
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // TODO: 考生自行补齐Tiling函数
    const gert::StorageShape* xShape =
        context->GetInputShape(0);


    const gert::Shape& shape =
        xShape->GetOriginShape();


    uint32_t totalLength = 1;


    for(size_t i = 0;
        i < shape.GetDimNum();
        i++)
    {
        totalLength *= shape.GetDim(i);
    }


    TanhCustomTilingData tilingData;


    tilingData.totalLength = totalLength;

    tilingData.tileNum = TILE_NUM;

    tilingData.tileLength =
        totalLength /
        BLOCK_DIM /
        TILE_NUM;



    context->SetBlockDim(BLOCK_DIM);



    memcpy(
        context->GetRawTilingData()->GetData(),
        &tilingData,
        sizeof(TanhCustomTilingData)
    );


    context->GetRawTilingData()
        ->SetDataSize(sizeof(TanhCustomTilingData));
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
