// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 本题输入固定为 (8, 2048)，总元素数为 16384。
    // 按照官方 Add 示例，固定使用 8 个 AI Core。
    constexpr uint32_t BLOCK_DIM = 8;
    constexpr uint32_t TILE_NUM = 8;

    // 获取输入 Tensor
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);

    // 获取输入数据类型
    ge::DataType dtype_x = tensor_x->GetDataType();

    // 获取输入元素个数
    uint32_t length_x = tensor_x->GetShapeSize();

    // 配置 Tiling Key
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取 Tiling 数据
    AddTilingData *tiling =
        context->GetTilingData<AddTilingData>();

    tiling->length = length_x;
    tiling->tile_num = TILE_NUM;

    // 固定使用 8 个 AI Core
    context->SetBlockDim(BLOCK_DIM);

    // 本算子不需要 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);

    // 输出 z 的 Shape 与输入 x 一致
    *z_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输出 z 的 dtype 与输入 x 一致
    const auto x_dtype = context->GetInputDataType(0);

    context->SetOutputDataType(0, x_dtype);

    return ge::GRAPH_SUCCESS;
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