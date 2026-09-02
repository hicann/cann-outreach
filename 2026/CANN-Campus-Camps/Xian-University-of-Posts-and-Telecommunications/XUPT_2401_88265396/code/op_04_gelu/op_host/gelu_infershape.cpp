/*!
 * \file gelu_infershape.cpp
 * \brief Gelu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGelu(
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

    /*
     * GELU 为逐元素算子：
     *
     * output.shape = input.shape
     */
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Gelu)
    .InferShape(InferShapeGelu);

} // namespace ops