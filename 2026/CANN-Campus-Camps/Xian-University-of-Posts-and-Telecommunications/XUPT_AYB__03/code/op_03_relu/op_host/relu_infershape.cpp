#include "register/op_impl_registry.h"

namespace ops {

ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    const gert::Shape* xShape =
        context->GetInputShape(0);

    gert::Shape* yShape =
        context->GetOutputShape(0);

    *yShape = *xShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu)
    .InferShape(InferShapeRelu);

}  // namespace ops