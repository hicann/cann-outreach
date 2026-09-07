#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取TilingData结构体对应的内存
    TanhCustomTilingData* tiling =
        context->GetTilingData<TanhCustomTilingData>();

    // 获取输入张量的总元素数量
    uint32_t totalLength =
        context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 设置参与运算的AI Core数量
    context->SetBlockDim(BLOCK_DIM);

    // 设置传递给Kernel侧的Tiling参数
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 本算子不需要额外的Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    // Tanh是逐元素算子，输出Shape和输入Shape一致
    *y_shape = *x1_shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);

    // 输出数据类型和输入数据类型一致
    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore()
            .AddConfig("ascend910b");
    }
};

OP_ADD(TanhCustom);

}  // namespace ops