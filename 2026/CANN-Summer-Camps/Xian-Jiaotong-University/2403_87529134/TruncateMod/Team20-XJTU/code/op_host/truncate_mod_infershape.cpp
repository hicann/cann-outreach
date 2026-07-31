/*!
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <vector>
#include <algorithm>
using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    //d TODO: 实现形状推导逻辑
    const gert::Shape* input_shapeX1 =context->GetInputShape(0);
    const gert::Shape* input_shapeX2 =context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape
    if (input_shapeX1 == nullptr || input_shapeX2 == nullptr) {
        outShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    size_t dimX1 = input_shapeX1->GetDimNum();
    size_t dimX2 = input_shapeX2->GetDimNum();
    size_t maxDim = std::max(dimX1, dimX2);
    //outShape->Clear();
    outShape->SetDimNum(0);
    std::vector<int64_t> tempDims;
    
for (size_t i = 0; i < maxDim; ++i) {
        int64_t lenX1 = 1;
        int64_t lenX2 = 1; 
        // 取当前维度下标，从尾部对齐
        size_t idxX1 = dimX1 - 1 - i;
        size_t idxX2 = dimX2 - 1 - i;
        if (i < dimX1) {
            lenX1 = input_shapeX1->GetDim(idxX1);
        }
        if (i < dimX2) {
            lenX2 = input_shapeX2->GetDim(idxX2);
        }

        bool invalid = (lenX1 != 1 && lenX2 != 1)
                && (lenX1 != lenX2)
                && (lenX1 != -1)
                && (lenX2 != -1);
        if (invalid) {
          return ge::GRAPH_PARAM_INVALID;
        }
        int64_t outLen;
        if (lenX1 == -1 || lenX2 == -1) {
          outLen = -1;
        } else {
          outLen = std::max(lenX1, lenX2);
        }
          tempDims.push_back(outLen);
}
  std::reverse(tempDims.begin(), tempDims.end());

for (auto dim : tempDims) {
    outShape->AppendDim(dim);
}
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);

} // namespace ops
