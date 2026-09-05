
// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {

const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取输入Tensor
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);

    // 获取输入x的数据类型，并选择对应的Tiling Key
    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取Tiling数据结构
    AddTilingData *tiling = context->GetTilingData<AddTilingData>();

    // 获取输入x的Shape
    const gert::StorageShape *x1_shape = context->GetInputShape(0);

    // 计算输入Tensor的总元素数量
    uint32_t data_sz = 1;
    for (int i = 0;
         i < x1_shape->GetStorageShape().GetDimNum();
         i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }

    // 写入Tiling参数
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    // 配置AI Core启动数量
    context->SetBlockDim(8);

    // 本算子不需要Workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // 获取输入x的Shape
    const gert::Shape *x1_shape = context->GetInputShape(0);

    // 获取输出z的Shape
    gert::Shape *z_shape = context->GetOutputShape(0);

    // 向量加法为逐元素运算，输出Shape与输入x一致
    *z_shape = *x1_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 获取输入x的数据类型
    const auto inputDataType = context->GetInputDataType(0);

    // 输出z的数据类型与输入x保持一致
    context->SetOutputDataType(0, inputDataType);

    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class Add : public OpDef {
public:
    explicit Add(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Add);

}  // namespace ops
