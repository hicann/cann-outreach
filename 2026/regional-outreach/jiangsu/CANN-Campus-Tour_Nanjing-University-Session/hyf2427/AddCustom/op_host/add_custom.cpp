/**
 * @file add_custom.cpp
 *
 * @brief add_custom 算子 host 侧实现：
 *        算子原型注册、shape 推导、数据类型推导、Tiling 实现
 *
 * 算子：z = x + y（逐元素加法）
 * 输入 x, y：2D [N2, N1]，float16，ND
 * 输出 z：shape 同输入
 */
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// AI Core 核数（块维度），需与 kernel 切分逻辑一致
const uint32_t BLOCK_DIM = 8;
// 每个核上的分块个数（tileNum * BUFFER_NUM 决定流水循环次数）
const uint32_t TILE_NUM = 8;

/**
 * Tiling 函数：根据输入 shape 计算切分参数并序列化 TilingData
 */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    // ND 格式下 2D 张量 [N2, N1] 按行主序连续存放，
    // GetShapeSize() 返回元素总数 N2 * N1，kernel 侧按一维处理
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 本算子不需要额外 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
/**
 * shape 推导：输出 shape 与输入 x 相同（逐元素运算，shape 不变）
 */
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);
    *z_shape = *x_shape;
    return GRAPH_SUCCESS;
}

/**
 * 数据类型推导：输出 dtype 与输入相同（float16）
 */
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
/**
 * 算子原型注册：定义 add_custom 的输入/输出属性并关联 Tiling 函数
 */
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        // 输入 x：必选，float16，ND
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输入 y：必选，float16，ND
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输出 z：必选，float16，ND
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 关联 shape / 数据类型推导函数
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // 关联 Tiling 函数，并注册算子支持的 AI 处理器型号
        // 注意：请根据实际硬件（npu-smi info 查询）保留对应型号
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910");
    }
};
// 结束算子注册
OP_ADD(AddCustom);
} // namespace ops
