/*!
 * \file sync_batch_norm_backward_reduce_infershape.cpp
 * \brief SyncBatchNormBackwardReduce 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t INPUT_NUM = 4;
static constexpr int64_t OUTPUT_NUM = 2;

static ge::graphStatus InferShapeSyncBatchNormBackwardReduce(gert::InferShapeContext* context)
{
    // 逐元素算子：所有输入 shape 相同，输出 shape 与输入相同
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    // 校验四路输入 shape 一致
    for (size_t i = 1; i < static_cast<size_t>(INPUT_NUM); i++) {
        const gert::Shape* curShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, curShape);
        OP_CHECK_IF(*curShape != *inputShape,
                    OP_LOGE(context, "input %zu shape mismatch with input 0", i),
                    return ge::GRAPH_FAILED);
    }

    // 两个输出 shape 与输入相同
    for (size_t i = 0; i < static_cast<size_t>(OUTPUT_NUM); i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBatchNormBackwardReduce).InferShape(InferShapeSyncBatchNormBackwardReduce);

} // namespace ops
