
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // TODO: 考生自行补齐Tiling函数
    TanhCustomTilingData tilingData;
    uint32_t totalLength = 1;

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::Shape storageShape = xShape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        totalLength *= storageShape.GetDim(i);
    }

    tilingData.totalLength = totalLength;
    tilingData.tileNum = TILE_NUM;

    context->SetBlockDim(BLOCK_DIM);
    size_t tilingSize = sizeof(TanhCustomTilingData);
    if (memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                 &tilingData, tilingSize) != EOK) {
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingSize);
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
