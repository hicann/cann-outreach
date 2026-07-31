/*!
 * \file truncate_div_infershape.cpp
 * \brief TruncateDiv 算子形状推导实现
 *
 * 实现两个输入的广播形状推导：
 * - 获取 x1 (input 0) 和 x2 (input 1) 的形状
 * - 按广播规则计算输出形状
 * - 如果有输入为空（shape = {}），则将其视为标量，维度为 0
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTruncateDiv(gert::InferShapeContext* context)
{
    // 获取两个输入的形状
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);

    if (x1Shape == nullptr || x2Shape == nullptr) {
        // 如果输入形状未知，设置为动态形状
        gert::Shape* outputShape = context->GetOutputShape(0);
        if (outputShape != nullptr) {
            // 如有至少一个已知输入，使用已知的
            const gert::Shape* knownShape = (x1Shape != nullptr) ? x1Shape : x2Shape;
            if (knownShape != nullptr) {
                *outputShape = *knownShape;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    // 获取输入维度
    size_t x1DimNum = x1Shape->GetDimNum();
    size_t x2DimNum = x2Shape->GetDimNum();

    // 如果两个输入都是标量 (dim=0)，输出也是标量
    if (x1DimNum == 0 && x2DimNum == 0) {
        gert::Shape* outputShape = context->GetOutputShape(0);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // 输出为标量，不设置任何维度
        *outputShape = gert::Shape();
        return ge::GRAPH_SUCCESS;
    }

    // 广播形状计算
    size_t broadcastDimNum = (x1DimNum > x2DimNum) ? x1DimNum : x2DimNum;

    // 将两个输入的维度补齐到 broadcastDimNum，缺失的前面补1
    std::vector<int64_t> x1Dims(broadcastDimNum, 1);
    std::vector<int64_t> x2Dims(broadcastDimNum, 1);
    std::vector<int64_t> outputDims(broadcastDimNum, 1);

    // 填充 x1 的维度（靠右对齐）
    for (size_t i = 0; i < x1DimNum; i++) {
        x1Dims[broadcastDimNum - x1DimNum + i] = x1Shape->GetDim(i);
    }

    // 填充 x2 的维度（靠右对齐）
    for (size_t i = 0; i < x2DimNum; i++) {
        x2Dims[broadcastDimNum - x2DimNum + i] = x2Shape->GetDim(i);
    }

    // 对每一维进行广播验证和合并
    for (size_t i = 0; i < broadcastDimNum; i++) {
        int64_t dim1 = x1Dims[i];
        int64_t dim2 = x2Dims[i];

        if (dim1 == dim2) {
            outputDims[i] = dim1;
        } else if (dim1 == 1) {
            outputDims[i] = dim2;
        } else if (dim2 == 1) {
            outputDims[i] = dim1;
        } else {
            // 维度不兼容，无法广播
            return ge::GRAPH_FAILED;
        }
    }

    // 设置输出形状（使用 SetDimNum/SetDim 逐维度设置）
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    outputShape->SetDimNum(broadcastDimNum);
    for (size_t i = 0; i < broadcastDimNum; i++) {
        outputShape->SetDim(i, outputDims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateDiv).InferShape(InferShapeTruncateDiv);

} // namespace ops
