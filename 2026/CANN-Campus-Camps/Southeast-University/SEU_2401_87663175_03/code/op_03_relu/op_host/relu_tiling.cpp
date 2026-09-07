/*!
 * \file relu_tiling.cpp
 * \brief Relu Tiling - low-overhead static Tensor strategy
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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TARGET_CORE_NUM = 16;
constexpr int64_t GM_ALIGN_BYTES = 512;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t platformCoreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(
        platformCoreNum <= 0,
        OP_LOGE(context, "invalid AIV core number"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "invalid UB size"),
        return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum must be positive"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();

    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "unsupported dtype"),
        return ge::GRAPH_FAILED);

    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t gmAlignNum = GM_ALIGN_BYTES / typeSize;

    // Never require the platform to have TARGET_CORE_NUM cores.
    // Use up to the target number, so this cannot reproduce the previous 561002
    // caused by a hard core-count condition.
    int64_t desiredCoreNum =
        (platformCoreNum < TARGET_CORE_NUM) ? platformCoreNum : TARGET_CORE_NUM;
    if (desiredCoreNum < 1) {
        desiredCoreNum = 1;
    }

    // 512B-aligned block starts improve GM movement efficiency.
    int64_t blockFactor = CeilDiv(totalNum, desiredCoreNum);
    blockFactor = CeilAlign(blockFactor, gmAlignNum);

    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum > platformCoreNum) {
        usedCoreNum = platformCoreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // Static Tensor mode uses a single in-place UB tensor instead of
    // input[2] + output[2], so almost the entire UB can be used by one tile.
    int64_t maxUbFactor =
        FloorAlign(static_cast<int64_t>(ubSize) / typeSize, gmAlignNum);
    OP_CHECK_IF(
        maxUbFactor <= 0,
        OP_LOGE(context, "UB is too small"),
        return ge::GRAPH_FAILED);

    int64_t ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
    if (ubFactor <= 0) {
        ubFactor = gmAlignNum;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    if (dataType == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling