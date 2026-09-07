/*!
 * \file square_infershape.cpp
 * \brief Square 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {

static ge::graphStatus InferShapeSquare(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape =
        context->GetInputShape(0);

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Square为逐元素算子，输出Shape与输入完全相同
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square)
    .InferShape(InferShapeSquare);

} // namespace ops