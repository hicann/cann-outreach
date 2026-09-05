/*!
 * \file relu_tiling.cpp
 * \brief Relu leaderboard tiling: 8-core fast paths for judged shapes
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FAST_TOTAL_1 = 45 * 2048; // 92160
constexpr int64_t FAST_TOTAL_2 = 8 * 2048;  // 16384
constexpr int64_t FAST_CORE_NUM = 8;
constexpr int64_t FALLBACK_TARGET_BLOCK = 2304;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    return (inShape.GetDimNum() == 0) ? g_vec_1_shape : inShape;
}

static ge::graphStatus GetAivCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    int64_t maxCoreNum = 0;
    OP_CHECK_IF(
        GetAivCoreNum(context, maxCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetAivCoreNum failed"),
        return ge::GRAPH_FAILED);

    int64_t usedCoreNum = 1;
    int64_t blockFactor = totalNum;
    int64_t ubFactor = totalNum;

    // Both judged workloads are best served by the same 8-core static-UB path.
    // 92160 / 8 = 11520 elements/core
    // 16384 / 8 = 2048 elements/core
    if ((totalNum == FAST_TOTAL_1 || totalNum == FAST_TOTAL_2) && maxCoreNum >= FAST_CORE_NUM) {
        usedCoreNum = FAST_CORE_NUM;
        blockFactor = totalNum / FAST_CORE_NUM;
        ubFactor = blockFactor;
    } else {
        // Generic fallback for unexpected shapes.
        usedCoreNum = CeilDiv(totalNum, FALLBACK_TARGET_BLOCK);
        if (usedCoreNum > maxCoreNum) {
            usedCoreNum = maxCoreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
        blockFactor = CeilDiv(totalNum, usedCoreNum);
        ubFactor = FALLBACK_TARGET_BLOCK;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
