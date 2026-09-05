/*!
 * \file relu_infershape.cpp
 * \brief Relu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRelu(
    gert::InferShapeContext* context)
{
    const gert::Shape* inputShape =
        context->GetInputShape(0);

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // ReLU 不改变 Tensor 的 shape
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu)
    .InferShape(InferShapeRelu);

} // namespace ops
