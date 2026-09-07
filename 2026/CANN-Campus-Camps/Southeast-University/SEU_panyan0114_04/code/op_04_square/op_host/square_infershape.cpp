/*!
 * \file square_infershape.cpp
 * \brief Square 绠楀瓙褰㈢姸鎺ㄥ瀹炵幇
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

    // Square is elementwise, so the output shape is identical to the input.
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Square).InferShape(InferShapeSquare);

} // namespace ops
