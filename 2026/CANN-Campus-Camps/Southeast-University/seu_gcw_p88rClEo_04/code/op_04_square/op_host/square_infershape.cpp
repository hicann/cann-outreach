/*!
 * \file square_infershape.cpp
 * \brief Square 算子形状和数据类型推导
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;

static ge::graphStatus InferShapeSquare(
    gert::InferShapeContext* context)
{
    const gert::Shape* inputShape =
        context->GetInputShape(INPUT_X_INDEX);

    gert::Shape* outputShape =
        context->GetOutputShape(OUTPUT_Y_INDEX);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Square 是元素级算子，完整复制输入 shape。
    //
    // 例如：
    // input  = [2, 3, 17]
    // output = [2, 3, 17]
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSquare(
    gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype =
        context->GetInputDataType(INPUT_X_INDEX);

    // 仅支持 float16 和 float32
    if (inputDtype != ge::DT_FLOAT16 &&
        inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // Square 不改变数据类型
    context->SetOutputDataType(
        OUTPUT_Y_INDEX,
        inputDtype);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square)
    .InferShape(InferShapeSquare)
    .InferDataType(InferDataTypeSquare);

} // namespace ops