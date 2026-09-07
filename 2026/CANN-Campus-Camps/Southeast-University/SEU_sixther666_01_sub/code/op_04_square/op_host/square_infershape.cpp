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
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSquare(gert::InferDataTypeContext* context)
{
    const ge::DataType inputType = context->GetInputDataType(0);
    if (inputType != ge::DT_FLOAT16 && inputType != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, inputType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square).InferShape(InferShapeSquare).InferDataType(InferDataTypeSquare);

} // namespace ops
