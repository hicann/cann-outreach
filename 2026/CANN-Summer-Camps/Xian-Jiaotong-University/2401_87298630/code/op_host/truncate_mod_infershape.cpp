/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    // 逐元素二元算子：输出 shape 为 x1、x2 的 NumPy 广播结果。
    // 等 shape 场景（验收主形态）下即等于输入 shape。
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    size_t n1 = x1_shape->GetDimNum();
    size_t n2 = x2_shape->GetDimNum();
    size_t n = (n1 > n2) ? n1 : n2;
    y_shape->SetDimNum(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t d1 = (i < n1) ? x1_shape->GetDim(n1 - 1 - i) : 1;
        int64_t d2 = (i < n2) ? x2_shape->GetDim(n2 - 1 - i) : 1;
        int64_t d = (d1 == 1) ? d2 : d1;  // 广播：非 1 维优先
        y_shape->SetDim(n - 1 - i, d);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);

} // namespace ops
