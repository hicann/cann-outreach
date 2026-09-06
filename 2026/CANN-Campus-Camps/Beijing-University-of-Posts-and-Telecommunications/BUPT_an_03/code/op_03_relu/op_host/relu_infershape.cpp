/*!
 * \file relu_infershape.cpp
 * \brief Relu 绠楀瓙褰㈢姸鎺ㄥ瀹炵幇
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    // TODO: 瀹炵幇褰㈢姸鎺ㄥ閫昏緫
    const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
    // 娉ㄦ剰锛氭棤杈撳叆绠楀瓙鏃?input_shape 涓?nullptr锛岄渶鍦ㄦ澶勬墜鍔ㄨ缃緭鍑?shape

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* in_shape = (i < 1) ? context->GetInputShape(i) : input_shape;
        if (in_shape == nullptr) {
            in_shape = input_shape;
        }
        if (in_shape != nullptr) {
            *output_shape = *in_shape;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops