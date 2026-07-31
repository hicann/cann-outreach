/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include <algorithm>
#include <vector>

using namespace ge;

namespace ops {

static const int64_t MAX_DIMS = 8;

static ge::graphStatus ComputeBroadcastShape(
    const gert::Shape* s1, const gert::Shape* s2,
    std::vector<int64_t>& outShape)
{
    int64_t d1 = static_cast<int64_t>(s1->GetDimNum());
    int64_t d2 = static_cast<int64_t>(s2->GetDimNum());
    int64_t rd = std::max(d1, d2);
    outShape.resize(rd, 1);

    for (int64_t i = 0; i < rd; ++i) {
        int64_t si1 = (i < rd - d1) ? 1 : s1->GetDim(static_cast<size_t>(i - (rd - d1)));
        int64_t si2 = (i < rd - d2) ? 1 : s2->GetDim(static_cast<size_t>(i - (rd - d2)));
        if (si1 != si2 && si1 != 1 && si2 != 1) {
            return ge::GRAPH_FAILED;
        }
        outShape[i] = std::max(si1, si2);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    const gert::Shape* sx = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sx);
    const gert::Shape* sy = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sy);

    gert::Shape* out = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);

    std::vector<int64_t> bcastShape;
    if (ComputeBroadcastShape(sx, sy, bcastShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(bcastShape.size());
    for (size_t i = 0; i < bcastShape.size(); ++i) {
        out->SetDim(i, bcastShape[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);

} // namespace ops
