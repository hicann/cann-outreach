
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取输入shape并计算总元素数
    const gert::StorageShape* x1_storage = context->GetInputShape(0);
    const gert::Shape& x1_shape = x1_storage->GetStorageShape();
    uint32_t totalLength = static_cast<uint32_t>(x1_shape.GetShapeSize());

    // 设置blockDim
    context->SetBlockDim(BLOCK_DIM);

    // 计算每个block处理的元素数
    // kernel中 loopCount = tileNum * BUFFER_NUM
    // tileLength = totalLength / tileNum
    // 需保证 loopCount * tileLength = lengthPerBlock
    // 因此 totalLength = lengthPerBlock / BUFFER_NUM
    uint32_t lengthPerBlock = totalLength / BLOCK_DIM;
    uint32_t tileNum = TILE_NUM;
    uint32_t tileLength = lengthPerBlock / (tileNum * BUFFER_NUM);

    // 设置Tiling参数
    auto* tilingData = context->GetTilingData<TanhCustomTilingData>();
    tilingData->totalLength = lengthPerBlock / BUFFER_NUM;
    tilingData->tileNum = tileNum;
    tilingData->tileLength = tileLength;

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
