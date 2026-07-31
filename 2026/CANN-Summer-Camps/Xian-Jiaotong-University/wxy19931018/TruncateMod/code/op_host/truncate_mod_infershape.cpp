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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);

    if (x1_shape == nullptr || x2_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 广播推导：计算 x1 和 x2 广播后的 shape
    size_t x1_dims = x1_shape->GetDimNum();
    size_t x2_dims = x2_shape->GetDimNum();
    size_t max_dims = std::max(x1_dims, x2_dims);

    std::vector<int64_t> result_dims(max_dims);

    for (size_t i = 0; i < max_dims; i++) {
        int64_t x1_dim = 1;
        int64_t x2_dim = 1;

        if (i < x1_dims) {
            x1_dim = x1_shape->GetDim(x1_dims - 1 - i);
        }
        if (i < x2_dims) {
            x2_dim = x2_shape->GetDim(x2_dims - 1 - i);
        }

        if (x1_dim == x2_dim) {
            result_dims[max_dims - 1 - i] = x1_dim;
        } else if (x1_dim == 1) {
            result_dims[max_dims - 1 - i] = x2_dim;
        } else if (x2_dim == 1) {
            result_dims[max_dims - 1 - i] = x1_dim;
        } else {
            // 广播规则不匹配
            return ge::GRAPH_FAILED;
        }
    }

    output_shape->SetDimNum(result_dims.size());
    for (size_t i = 0; i < result_dims.size(); ++i) {
    output_shape->SetDim(i, result_dims[i]);
  }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);

} // namespace ops
