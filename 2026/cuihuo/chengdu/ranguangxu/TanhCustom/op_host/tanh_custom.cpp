
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
// 核（Block）数量与单核内 tile 数量，均为编译期常量。
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取输入张量总元素个数，并按 BLOCK_DIM 平均切分到每个核。
    // 注意：此处为整除，若 totalLength 不能被 BLOCK_DIM 整除，余数元素将被丢弃（保持原有行为）。
    const uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto* tilingData = context->GetTilingData<TanhCustomTilingData>();
    tilingData->blockLength = totalLength / BLOCK_DIM;
    tilingData->tileNum = TILE_NUM;
    context->SetBlockDim(BLOCK_DIM);

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(TanhCustom);
}
