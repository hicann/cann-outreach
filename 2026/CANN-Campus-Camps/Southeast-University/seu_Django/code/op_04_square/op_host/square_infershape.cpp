/*!
 * \file square_infershape.cpp
 * \brief Output shape is identical to input shape, including scalars.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static ge::graphStatus InferShapeSquare(gert::InferShapeContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    auto* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square).InferShape(InferShapeSquare);
} // namespace ops
