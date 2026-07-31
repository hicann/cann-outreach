/*!
 * \file sync_batch_norm_backward_reduce_infershape.cpp
 * \brief SyncBatchNormBackwardReduce 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSyncBatchNormBackwardReduce(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* sumDyXmuShape = context->GetOutputShape(0);
    gert::Shape* yShape = context->GetOutputShape(1);
    if (inputShape == nullptr || sumDyXmuShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *sumDyXmuShape = *inputShape;
    *yShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBatchNormBackwardReduce).InferShape(InferShapeSyncBatchNormBackwardReduce);

} // namespace ops
