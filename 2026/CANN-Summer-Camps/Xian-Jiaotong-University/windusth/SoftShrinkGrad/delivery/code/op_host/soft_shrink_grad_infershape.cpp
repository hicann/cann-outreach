/*!
 * \file soft_shrink_grad_infershape.cpp
 * \brief SoftShrinkGrad 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSoftShrinkGrad(gert::InferShapeContext* context)
{
    const gert::Shape* gradShape = context->GetInputShape(0);
    const gert::Shape* xShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (gradShape == nullptr || xShape == nullptr || yShape == nullptr || *gradShape != *xShape) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *gradShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftShrinkGrad).InferShape(InferShapeSoftShrinkGrad);

} // namespace ops
