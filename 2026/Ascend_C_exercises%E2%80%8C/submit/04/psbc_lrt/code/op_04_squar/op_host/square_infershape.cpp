/*!
 * \file square_infershape.cpp
 * \brief Square 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSquare(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (input_shape == nullptr || output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square).InferShape(InferShapeSquare);

} // namespace ops