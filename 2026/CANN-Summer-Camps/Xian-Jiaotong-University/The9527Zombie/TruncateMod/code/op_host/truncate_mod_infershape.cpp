/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod shape / dtype inference.
 *
 * Output y takes the NumPy-style broadcast shape of x1 and x2, and the dtype
 * of x1 (x1 and x2 must share the same dtype). Two dims are compatible when
 * they are equal or one of them is 1; the output dim is the larger one.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_X1_IDX = 0;
static constexpr size_t INPUT_X2_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const size_t rank1 = x1Shape->GetDimNum();
    const size_t rank2 = x2Shape->GetDimNum();
    const size_t outRank = (rank1 > rank2) ? rank1 : rank2;

    yShape->SetDimNum(outRank);
    // Align both shapes to the right; missing high dims are treated as 1.
    for (size_t i = 0; i < outRank; ++i) {
        const int64_t d1 = (i < outRank - rank1) ? 1 : x1Shape->GetDim(i - (outRank - rank1));
        const int64_t d2 = (i < outRank - rank2) ? 1 : x2Shape->GetDim(i - (outRank - rank2));
        int64_t outDim;
        if (d1 == d2 || d2 == 1) {
            outDim = d1;
        } else if (d1 == 1) {
            outDim = d2;
        } else {
            OP_LOGE(context, "TruncateMod: dim %zu can not broadcast (x1=%ld, x2=%ld).", i, d1, d2);
            return ge::GRAPH_FAILED;
        }
        yShape->SetDim(i, outDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTruncateMod(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_X1_IDX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod).InferDataType(InferDataTypeTruncateMod);
} // namespace ops
