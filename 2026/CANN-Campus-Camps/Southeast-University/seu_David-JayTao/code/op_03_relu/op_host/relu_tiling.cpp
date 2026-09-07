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

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
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
    uint64_t ubSize;
    int64_t coreNum;

    // ============================================================
    // 1. 获取硬件资源
    // ============================================================
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 2. 获取输入 shape
    // ============================================================
    const gert::StorageShape* inputShape = context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape shape =
        EnsureNotScalar(inputShape->GetStorageShape());

    int64_t totalNum = shape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum is invalid"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 3. 获取 TilingData
    // ============================================================
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;

    // ============================================================
    // 4. 核间切分
    //
    // 数据很小时没必要强行把所有 Core 拉起来
    // 大数据时尽量使用更多 Core
    // ============================================================
    int64_t usedCoreNum = coreNum;

    int64_t preferredCoreNum =
        CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);

    if (preferredCoreNum < 1) {
        preferredCoreNum = 1;
    }

    if (preferredCoreNum < usedCoreNum) {
        usedCoreNum = preferredCoreNum;
    }

    tiling->blockFactor =
        CeilDiv(totalNum, usedCoreNum);

    // 重新计算实际需要启动的核数
    usedCoreNum =
        CeilDiv(totalNum, tiling->blockFactor);

    context->SetBlockDim(usedCoreNum);

    // ============================================================
    // 5. 核内 UB 切分
    //
    // ReLU:
    // inputQueue  2 buffers
    // outputQueue 2 buffers
    //
    // 共 4 块 UB buffer
    //
    // TYPE_SIZE=4 按 fp32 的最坏情况估算，
    // 对 fp16 只是稍微保守一些，但安全。
    // ============================================================
    constexpr int64_t UB_BUFFER_COUNT = 4;

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    int64_t ubCanUse =
        static_cast<int64_t>(ubSize);

    int64_t maxUbFactor =
        FloorDiv(
            FloorDiv(ubCanUse, TYPE_SIZE),
            UB_BUFFER_COUNT);

    tiling->ubFactor =
        FloorAlign(maxUbFactor, ubBlockSize);

    // 没必要一次 UB tile 比整个 block 还大
    if (tiling->ubFactor > tiling->blockFactor) {
        tiling->ubFactor = tiling->blockFactor;
    }

    OP_CHECK_IF(
        tiling->ubFactor <= 0,
        OP_LOGE(context, "ubFactor is invalid"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 6. 根据 dtype 选择 Kernel 模板
    // ============================================================
    uint64_t tilingKey;

    auto inputDesc = context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
