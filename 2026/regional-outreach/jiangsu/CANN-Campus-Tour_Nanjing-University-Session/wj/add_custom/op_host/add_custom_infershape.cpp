/*!
 * \file add_custom_infershape.cpp
 * \brief add_custom 算子的 shape 推理和数据类型推理实现
 *
 * 本文件提供推理逻辑，用于确定 add_custom 算子的输出张量 shape 和数据类型。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

// 常量索引定义
static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 推理 add_custom 算子的输出 shape
 *
 * 获取输入张量 x 的 shape 并传播到输出张量 z。
 * 对于 add_custom 算子，输出 shape 与输入 shape 相同。
 *
 * @param context 指向 shape 推理上下文的指针
 * @return 推理成功返回 ge::GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferShapeAddCustom(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAddCustom");

    // 获取输入 x 的 shape 信息
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 获取输出 z 的 shape 信息
    gert::Shape* zShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);

    // 填充输出 shape 的维度和大小（add_custom 输出 shape 与输入 x 相同）
    auto xShapeSize = xShape->GetDimNum();
    zShape->SetDimNum(xShapeSize);

    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        zShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAddCustom");
    return GRAPH_SUCCESS;
}

/*!
 * \brief 推理 add_custom 算子的输出数据类型
 *
 * 获取输入张量 x 的数据类型并传播到输出张量 z。
 * 对于 add_custom 算子，输出数据类型与输入数据类型相同。
 *
 * @param context 指向数据类型推理上下文的指针
 * @return 推理成功返回 ge::GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferDataTypeAddCustom(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddCustom");

    // 设置输出的数据类型（与输入 x 相同）
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddCustom");
    return GRAPH_SUCCESS;
}

// infershape 注册入口：将 shape 推理函数和数据类型推理函数注册到系统中
IMPL_OP_INFERSHAPE(AddCustom).InferShape(InferShapeAddCustom).InferDataType(InferDataTypeAddCustom);
} // namespace ops
