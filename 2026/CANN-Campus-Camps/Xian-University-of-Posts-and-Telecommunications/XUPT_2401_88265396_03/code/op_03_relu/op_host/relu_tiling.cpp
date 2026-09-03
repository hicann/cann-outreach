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
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
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
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    /*
     * 获取真实输入元素数量。
     *
     * 不直接写死题面中的 (8, 2048)，因为工程自带测试还存在
     * (45, 2048) 等 shape。
     */
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape storageShape =
        EnsureNotScalar(inputShape->GetStorageShape());

    const int64_t totalNum = storageShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "invalid totalNum"),
        return ge::GRAPH_FAILED);

    /*
     * 获取 dtype。
     */
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType dataType = inputDesc->GetDataType();

    /*
     * float32 : 4 Byte
     * float16 : 2 Byte
     *
     * 原模板同时判断了 BF16，这里保持原来的兼容逻辑。
     */
    int64_t typeSize = TYPE_SIZE;

    if (dataType == ge::DT_FLOAT16 ||
        dataType == ge::DT_BF16) {
        typeSize = 2;
    }

    /*
     * Ascend 数据搬运的基本 datablock 为 32 Byte。
     *
     * float32:
     *     32 / 4 = 8 elements
     *
     * float16:
     *     32 / 2 = 16 elements
     */
    constexpr int64_t BLOCK_BYTES = 32;

    const int64_t alignNum =
        BLOCK_BYTES / typeSize;

    /*
     * 多核切分。
     *
     * 每个 AIV Core 目标处理至少约 1024 个元素，
     * 避免输入不大时启用过多 Core 带来的调度开销。
     */
    int64_t usedCoreNum =
        (totalNum + MIN_SPLIT_THRESHOLD - 1) /
        MIN_SPLIT_THRESHOLD;

    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    /*
     * 向下寻找：
     *
     * 1. totalNum 可以被 core 数整除
     * 2. 每个 Core 的 blockFactor 满足 32B 对齐
     *
     * 这样 Kernel 不需要处理 core remainder。
     */
    while (usedCoreNum > 1) {
        if ((totalNum % usedCoreNum == 0) &&
            (((totalNum / usedCoreNum) % alignNum) == 0)) {
            break;
        }

        --usedCoreNum;
    }

    const int64_t blockFactor =
        totalNum / usedCoreNum;

    /*
     * UB 切分。
     *
     * Kernel 中：
     *
     * input queue  × double buffer
     * output queue × double buffer
     *
     * 共需要 4 份 tile 空间。
     *
     * 同时预留 8 KB UB，避免把整个 UB 全部分配掉。
     */
    constexpr uint64_t UB_RESERVED_BYTES =
        8 * 1024;

    const uint64_t usableUb =
        (ubSize > UB_RESERVED_BYTES)
            ? (ubSize - UB_RESERVED_BYTES)
            : ubSize;

    int64_t maxUbFactor =
        static_cast<int64_t>(
            usableUb /
            (4ULL * static_cast<uint64_t>(typeSize)));

    /*
     * UB tile 也保持 32B 对齐。
     */
    maxUbFactor =
        (maxUbFactor / alignNum) * alignNum;

    if (maxUbFactor < alignNum) {
        maxUbFactor = alignNum;
    }

    int64_t ubFactor =
        (blockFactor < maxUbFactor)
            ? blockFactor
            : maxUbFactor;

    ubFactor =
        (ubFactor / alignNum) * alignNum;

    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }

    /*
     * 写入 TilingData。
     */
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    /*
     * 设置实际启动的 AIV Core 数。
     */
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    /*
     * 根据输入 dtype 选择 tilingKey。
     *
     * mode 0 -> half
     * mode 1 -> float
     *
     * 与原始 relu.cpp 保持完全一致。
     */
    uint64_t tilingKey;

    if (dataType == ge::DT_FLOAT16 ||
        dataType == ge::DT_BF16) {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling