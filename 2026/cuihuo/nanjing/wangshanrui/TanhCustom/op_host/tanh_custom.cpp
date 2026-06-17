
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // TODO: 考生自行补齐Tiling函数
    // 获取输入tensor的StorageShape
    const gert::StorageShape* x_storage_shape = context->GetInputShape(0);
    
    // 从StorageShape获取实际Shape
    const gert::Shape& x_shape = x_storage_shape->GetShape();
    
    // 手动计算总元素数
    uint32_t totalElements = 1;
    for (int i = 0; i < x_shape.GetDimNum(); i++) {
        totalElements *= x_shape.GetDim(i);
    }
    
    // 计算每个block处理的元素数
    uint32_t elementsPerBlock = (totalElements + BLOCK_DIM - 1) / BLOCK_DIM;
    
    // 设置block维度
    context->SetBlockDim(BLOCK_DIM);
    
    // 通过GetTilingData获取指针并填充
    auto* tilingData = context->GetTilingData<TanhCustomTilingData>();
    tilingData->totalElements = totalElements;
    tilingData->elementsPerBlock = elementsPerBlock;
    
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
