/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    // 输入 dtype 一致性校验（DESIGN.md §3.1）：
    // op_def 中 x1/x2 的 dtype 集合各自独立（{BF16, FP16, FP32}），若混合 dtype
    // （如 x1=FP16、x2=FP32），tilingKey 仅按 x1 dtype 分派会以错误步长解析输入，
    // 静默产出错误结果，必须在此处前置校验并拒绝。
    const gert::Tensor* x1Tensor = context->GetInputTensor(0);
    const gert::Tensor* x2Tensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Tensor);
    if (x1Tensor->GetDataType() != x2Tensor->GetDataType()) {
        OP_LOGE(context, "TruncateMod inputs x1/x2 dtype mismatch: x1=%d x2=%d",
                x1Tensor->GetDataType(), x2Tensor->GetDataType());
        return ge::GRAPH_FAILED;
    }

    // rank-0 标量约定（DESIGN.md §3.1）：沿用骨架 EnsureNotScalar 语义，
    // 0-d 输入统一按 {1} 处理（与 numpy 0-d→0-d 语义有偏差，为既有骨架行为延续）。
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    size_t rankA = x1Shape->GetDimNum();
    size_t rankB = x2Shape->GetDimNum();
    size_t rankOut = (rankA > rankB) ? rankA : rankB;

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetDimNum(rankOut);

    // numpy 右对齐广播：低 rank 左侧补 1，逐维取 max（除 1 与不同维冲突时非法）
    for (size_t d = 0; d < rankOut; d++) {
        int64_t dimA = 1;
        int64_t dimB = 1;
        if (rankA > 0 && d >= rankOut - rankA) {
            dimA = x1Shape->GetDim(d - (rankOut - rankA));
        }
        if (rankB > 0 && d >= rankOut - rankB) {
            dimB = x2Shape->GetDim(d - (rankOut - rankB));
        }
        if (dimA == dimB) {
            outputShape->SetDim(d, dimA);
        } else if (dimA == 1) {
            outputShape->SetDim(d, dimB);
        } else if (dimB == 1) {
            outputShape->SetDim(d, dimA);
        } else {
            OP_LOGE(context, "TruncateMod broadcast invalid at dim %zu: x1=%ld x2=%ld", d, dimA, dimB);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);

} // namespace ops
