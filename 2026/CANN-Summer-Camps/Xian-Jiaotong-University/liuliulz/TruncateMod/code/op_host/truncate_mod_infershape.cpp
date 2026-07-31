/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ge {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* s1 = context->GetInputShape(0);
    const gert::Shape* s2 = context->GetInputShape(1);
    gert::Shape* sy = context->GetOutputShape(0);
    if (sy == nullptr) return GRAPH_FAILED;
    if (s1 == nullptr && s2 == nullptr) return GRAPH_FAILED;
    if (s1 == nullptr) { *sy = *s2; return GRAPH_SUCCESS; }
    if (s2 == nullptr) { *sy = *s1; return GRAPH_SUCCESS; }

    int64_t r1 = s1->GetDimNum(), r2 = s2->GetDimNum();
    int64_t r = (r1 > r2) ? r1 : r2;
    std::vector<int64_t> out(r, 1);
    for (int64_t i = 0; i < r; i++) {
        int64_t d1 = (i < r1) ? s1->GetDim(r1 - 1 - i) : 1;
        int64_t d2 = (i < r2) ? s2->GetDim(r2 - 1 - i) : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return GRAPH_FAILED;
        out[r - 1 - i] = (d1 > d2) ? d1 : d2;
    }
    sy->SetDimNum(r);
    for (int64_t i = 0; i < r; i++) sy->SetDim(i, out[i]);
    return GRAPH_SUCCESS;
}
ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(TruncateMod).InferShape(ge::InferShape);
} // namespace ops
