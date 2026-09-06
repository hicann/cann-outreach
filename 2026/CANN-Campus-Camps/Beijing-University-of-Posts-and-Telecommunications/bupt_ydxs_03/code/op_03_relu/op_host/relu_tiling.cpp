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

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return in_shape;
}

/*
 * 获取平台信息
 */
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(
            context,
            "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 获取 Workspace 大小
 */
static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

/*
 * ReLU Tiling
 */
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // ============================================================
    // 1. 获取平台信息
    // ============================================================

    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 2. 设置 Workspace
    // ============================================================

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 3. 获取 TilingData
    // ============================================================

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    // ============================================================
    // 4. 获取输入 Shape
    // ============================================================

    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    /*
     * CANN 9.0.0 中 GetInputShape()
     * 返回的是 StorageShape。
     *
     * StorageShape 需要通过 GetShape()
     * 获取真正的 gert::Shape。
     */
    const gert::Shape& shape =
        inputShape->GetShape();

    // ============================================================
    // 5. 计算输入 Tensor 的元素总数
    // ============================================================

    int64_t totalNum = 1;

    for (size_t i = 0;
         i < shape.GetDimNum();
         ++i) {

        totalNum *= shape.GetDim(i);
    }

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "totalNum is invalid"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 6. 获取输入数据类型
    // ============================================================

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();

    // ============================================================
    // 7. 确定元素大小
    // ============================================================

    int64_t typeSize = 0;

    if (dataType == ge::DT_FLOAT16) {

        typeSize = 2;

    } else if (dataType == ge::DT_FLOAT) {

        typeSize = 4;

    } else if (dataType == ge::DT_BF16) {

        typeSize = 2;

    } else {

        OP_LOGE(
            context,
            "Unsupported data type");

        return ge::GRAPH_FAILED;
    }

    // ============================================================
    // 8. 计算 UB Tile 大小
    // ============================================================

    /*
     * Kernel 中使用：
     *
     * BUFFER_NUM = 2
     *
     * 输入：
     *     2 个 Buffer
     *
     * 输出：
     *     2 个 Buffer
     *
     * 因此需要考虑 4 份 UB Buffer。
     *
     * 这里使用约 3/4 UB 空间进行计算，
     * 剩余空间用于运行时开销。
     */

    uint64_t usableUb =
        ubSize * 3 / 4;

    uint64_t ubBytes =
        usableUb / 4;

    int64_t ubFactor =
        static_cast<int64_t>(
            ubBytes /
            static_cast<uint64_t>(typeSize));

    // ============================================================
    // 9. 按 UB Block 大小进行对齐
    // ============================================================

    uint32_t ubBlockSize =
        GetUbBlockSize(context);

    if (ubBlockSize == 0) {
        ubBlockSize = 32;
    }

    int64_t alignNum =
        static_cast<int64_t>(
            ubBlockSize /
            static_cast<uint32_t>(typeSize));

    if (alignNum > 0) {

        ubFactor =
            FloorAlign(
                ubFactor,
                alignNum);
    }

    // ============================================================
    // 10. 防止 UB Tile 为 0
    // ============================================================

    if (ubFactor <= 0) {

        ubFactor =
            alignNum > 0
                ? alignNum
                : 1;
    }

    // ============================================================
    // 11. UB Tile 不超过总元素数量
    // ============================================================

    if (ubFactor > totalNum) {
        ubFactor = totalNum;
    }

    // ============================================================
    // 12. 确定 Block 数
    // ============================================================

    /*
     * 本题输入固定为：
     *
     *     (8, 2048)
     *
     * 总元素：
     *
     *     16384
     *
     * Ascend 910B 有多个 AI Core。
     *
     * 对 ElementWise ReLU，
     * 让多个 AI Core 并行处理数据。
     *
     * 优先选择能够整除 totalNum 的 Core 数。
     */

    int64_t blockNum =
        coreNum;

    if (blockNum > totalNum) {
        blockNum = totalNum;
    }

    /*
     * 找到不超过 coreNum 的、
     * 能够整除 totalNum 的最大 Block 数。
     */
    while (blockNum > 1 &&
           totalNum % blockNum != 0) {

        --blockNum;
    }

    if (blockNum <= 0) {
        blockNum = 1;
    }

    // ============================================================
    // 13. 计算每个 Block 的数据量
    // ============================================================

    int64_t blockFactor =
        totalNum / blockNum;

    OP_CHECK_IF(
        blockFactor <= 0,
        OP_LOGE(
            context,
            "blockFactor is invalid"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 14. 如果 Block 数据量小于 UB Tile，
    //     缩小 UB Tile
    // ============================================================

    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // ============================================================
    // 15. 写入 TilingData
    // ============================================================

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;

    // ============================================================
    // 16. 设置 BlockDim
    // ============================================================

    context->SetBlockDim(
        static_cast<uint32_t>(blockNum));

    // ============================================================
    // 17. 根据 dtype 设置 TilingKey
    // ============================================================

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

    context->SetTilingKey(
        tilingKey);

    return ge::GRAPH_SUCCESS;
}

/*
 * Tiling Parse
 */
static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

/*
 * Compile Info
 */
struct ReluCompileInfo {};

/*
 * 注册 ReLU Tiling
 */
IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

}  // namespace optiling