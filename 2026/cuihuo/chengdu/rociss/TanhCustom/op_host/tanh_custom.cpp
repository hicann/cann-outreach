
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
constexpr uint32_t BUFFER_NUM = 2; // 与kernel侧BUFFER_NUM保持一致
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
// 1. 获取输入shape信息
   
     const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = static_cast<uint32_t>(xShape->GetStorageShape().GetShapeSize());

    // 2. 设置tiling数据：核间均分，核内按tile切分
    TanhCustomTilingData* tiling = context->GetTilingData<TanhCustomTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = totalLength;
    tiling->blockLength = totalLength / BLOCK_DIM;
    tiling->tileNum = TILE_NUM;
    tiling->tileLength = tiling->blockLength / tiling->tileNum / BUFFER_NUM;

    // 3. 设置核数
    context->SetBlockDim(BLOCK_DIM);
    // TODO: 考生自行补齐Tiling函数
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
