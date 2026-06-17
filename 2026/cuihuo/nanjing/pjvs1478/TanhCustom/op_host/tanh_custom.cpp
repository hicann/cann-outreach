
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TanhCustomTilingData tiling;

    // 获取输入tensor的shape信息，计算总元素个数
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalElements = static_cast<uint32_t>(x_shape->GetShape().GetShapeSize());

    // 计算每个核处理的元素个数
    uint32_t perCoreLength = totalElements / BLOCK_DIM;

    // 设置tiling参数
    tiling.totalLength = perCoreLength;
    tiling.tileNum = TILE_NUM;

    // 设置Block维度（核数）
    context->SetBlockDim(BLOCK_DIM);

    // 将tiling数据写入context
    auto tilingData = context->GetTilingData<TanhCustomTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *tilingData = tiling;

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
