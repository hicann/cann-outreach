/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_SIZE = 32;              // GM/UB 搬运基本对齐粒度，Byte
constexpr int64_t BUFFER_NUM = 2;               // DoubleBuffer
constexpr int64_t QUEUE_NUM = 2;                // inputQueueX + outputQueueY
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;   // 小数据不必过度拆核
constexpr uint64_t UB_RESERVED_SIZE = 8 * 1024; // 给系统/临时开销预留少量 UB

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "coreNum is invalid"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t maxCoreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            maxCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // =========================================================
    // 1. 获取输入 Shape
    // =========================================================

    const gert::StorageShape* xStorageShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        xStorageShape);

    const gert::Shape xShape =
        EnsureNotScalar(
            xStorageShape->GetStorageShape());

    const int64_t totalNum =
        xShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(context, "invalid input shape"),
        return ge::GRAPH_FAILED);

    // =========================================================
    // 2. 获取输入数据类型
    // =========================================================

    const auto* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    const ge::DataType dataType =
        inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16) {
        typeSize = 2;

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);

    } else if (dataType == ge::DT_FLOAT) {
        typeSize = 4;

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);

    } else {
        OP_LOGE(
            context,
            "Relu only supports float16 and float "
            "in current op definition");

        return ge::GRAPH_FAILED;
    }

    // 每个 32 Byte 对应多少个元素
    const int64_t alignNum =
        BLOCK_SIZE / typeSize;

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    // =========================================================
    // 3. 空 Tensor
    // =========================================================

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;

        // 保证 Kernel InitBuffer 参数合法
        tiling->ubFactor = alignNum;

        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);

        return ge::GRAPH_SUCCESS;
    }

    // =========================================================
    // 4. 多核切分
    // =========================================================
    //
    // 小数据不宜启动过多 AIV Core；
    // 当数据较大时，尽可能利用更多 Vector Core。
    //
    // MIN_SPLIT_THRESHOLD = 1024：
    // 大致保证每增加一个 Core 至少有约 1024 个元素。
    // =========================================================

    int64_t useCoreNum =
        totalNum / MIN_SPLIT_THRESHOLD;

    if (useCoreNum < 1) {
        useCoreNum = 1;
    }

    if (useCoreNum > maxCoreNum) {
        useCoreNum = maxCoreNum;
    }

    if (useCoreNum > totalNum) {
        useCoreNum = totalNum;
    }

    // 每个核最多处理的元素数量
    int64_t blockFactor =
        (totalNum + useCoreNum - 1)
        / useCoreNum;

    // ---------------------------------------------------------
    // blockFactor 按 32 Byte 对齐
    //
    // 这样：
    //
    // Core 0 -> offset = 0
    // Core 1 -> offset = blockFactor
    // Core 2 -> offset = 2 * blockFactor
    //
    // 每个核的 GM 起始地址都保持 32 Byte 对齐。
    // ---------------------------------------------------------

    blockFactor =
        ((blockFactor + alignNum - 1)
            / alignNum)
        * alignNum;

    // 对齐后重新计算真正需要启动的核数，
    // 防止出现完全不工作的空 Core。
    useCoreNum =
        (totalNum + blockFactor - 1)
        / blockFactor;

    // =========================================================
    // 5. UB Tiling + DoubleBuffer
    // =========================================================
    //
    // Kernel 中存在：
    //
    // inputQueueX  : BUFFER_NUM = 2
    // outputQueueY : BUFFER_NUM = 2
    //
    // 因此总共有：
    //
    // 2 * 2 = 4
    //
    // 个等大小的 UB Buffer。
    //
    // 每个 Buffer 可以使用：
    //
    // usableUbSize / 4
    //
    // =========================================================

    uint64_t usableUbSize = ubSize;

    if (usableUbSize > UB_RESERVED_SIZE) {
        usableUbSize -= UB_RESERVED_SIZE;
    }

    const uint64_t oneBufferBytes =
        usableUbSize
        / (QUEUE_NUM * BUFFER_NUM);

    int64_t maxUbFactor =
        static_cast<int64_t>(
            oneBufferBytes
            / static_cast<uint64_t>(typeSize));

    // UB Tile 同样按照 32 Byte 对齐
    maxUbFactor =
        (maxUbFactor / alignNum)
        * alignNum;

    OP_CHECK_IF(
        maxUbFactor <= 0,
        OP_LOGE(context, "UB is too small"),
        return ge::GRAPH_FAILED);

    // 如果单核数据能够一次全部放入 UB，
    // 就没有必要进一步拆 Tile。
    int64_t ubFactor =
        blockFactor < maxUbFactor
            ? blockFactor
            : maxUbFactor;

    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // =========================================================
    // 6. 写回 TilingData
    // =========================================================

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;

    // =========================================================
    // 7. 设置 BlockDim / TilingKey
    // =========================================================

    context->SetBlockDim(
        static_cast<uint32_t>(useCoreNum));

    context->SetTilingKey(
        tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling