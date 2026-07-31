/*!
 * \file selu_grad_infershape.cpp
 * \brief SeluGrad 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSeluGrad(gert::InferShapeContext* context)
{
    const gert::Shape* gradientsShape = context->GetInputShape(0);
    const gert::Shape* outputsShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (gradientsShape == nullptr || outputsShape == nullptr || yShape == nullptr || *gradientsShape != *outputsShape) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *gradientsShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SeluGrad).InferShape(InferShapeSeluGrad);

} // namespace ops
