
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // TODO: 考生自行补齐Tiling函数
	    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    // 从StorageShape中获取实际的Shape
    const gert::Shape* inputShape = &(inputStorageShape->GetOriginShape());
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    // 2. 计算总数据量
    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape->GetDimNum(); i++) {
        totalNum *= inputShape->GetDim(i);
    }
    
    // 3. 设置tiling参数
    TanhCustomTilingData* tilingData = context->GetTilingData<TanhCustomTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    tilingData->totalNum = static_cast<uint32_t>(totalNum);
    tilingData->blockNum = BLOCK_DIM * TILE_NUM;
    tilingData->dataPerBlock = static_cast<uint32_t>(totalNum / (BLOCK_DIM * TILE_NUM) + 1);
    
    // 4. 设置核数
    context->SetBlockDim(BLOCK_DIM * TILE_NUM);
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
