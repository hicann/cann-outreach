/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子固定八核 Tiling
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr int64_t BLOCK_DIM = 8;

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* inputShape =
        context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum =
        inputShape->GetStorageShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0 ||
            totalNum % BLOCK_DIM != 0,
        OP_LOGE(
            context,
            "input size must be divisible by 8"),
        return ge::GRAPH_FAILED);

    const auto* inputDesc =
        context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType dataType =
        inputDesc->GetDataType();

    OP_CHECK_IF(
        dataType != ge::DT_FLOAT16 &&
            dataType != ge::DT_FLOAT,
        OP_LOGE(
            context,
            "only float16 and float32 are supported"),
        return ge::GRAPH_FAILED);

    const int64_t blockFactor =
        totalNum / BLOCK_DIM;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = blockFactor;

    context->SetBlockDim(BLOCK_DIM);

    size_t* workspace =
        context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    const uint64_t tilingKey =
        dataType == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(
                  RELU_TPL_SCH_MODE_0)
            : GET_TPL_TILING_KEY(
                  RELU_TPL_SCH_MODE_1);

    context->SetTilingKey(tilingKey);

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