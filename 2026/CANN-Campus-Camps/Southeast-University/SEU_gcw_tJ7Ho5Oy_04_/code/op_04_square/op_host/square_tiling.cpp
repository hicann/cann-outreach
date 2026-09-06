/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 512;

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

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    const gert::Tensor* inputTensor =
        context->GetRequiredInputTensor(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputTensor);

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    int64_t totalNum =
        static_cast<int64_t>(
            inputTensor->GetShapeSize());

    int64_t typeSize =
        inputDesc->GetDataType() == ge::DT_FLOAT16
            ? 2
            : 4;

    int64_t alignElements =
        32 / typeSize;

    int64_t blockDim =
        CeilDiv(
            totalNum,
            MIN_SPLIT_THRESHOLD);

    if (blockDim < 1) {
        blockDim = 1;
    }

    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    if (blockDim > 32) {
        blockDim = 32;
    }

    int64_t blockFactor =
        CeilDiv(totalNum, blockDim);

    int64_t blockRemainder =
        blockFactor % alignElements;

    if (blockRemainder != 0) {
        blockFactor +=
            alignElements - blockRemainder;
    }

    int64_t maxUbFactor =
        static_cast<int64_t>(ubSize) /
        typeSize /
        2;

    maxUbFactor =
        FloorAlign(
            maxUbFactor,
            alignElements);

    int64_t ubFactor =
        blockFactor < maxUbFactor
            ? blockFactor
            : maxUbFactor;

    if (ubFactor < alignElements) {
        ubFactor = alignElements;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

    uint64_t tilingKey;

    if (inputDesc->GetDataType() ==
        ge::DT_FLOAT16) {
        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

}