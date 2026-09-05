/*!
 * \file square_infershape.cpp
 * \brief Square 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSquare(
    gert::InferShapeContext* context)
{
    const gert::Shape* inputShape =
        context->GetInputShape(0);

    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Square 不改变 Tensor 的 Shape
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square)
    .InferShape(InferShapeSquare);

} // namespace ops