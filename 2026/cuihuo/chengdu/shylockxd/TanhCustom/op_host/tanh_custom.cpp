
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstring>


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TanhCustomTilingData tilingData;

    // 1. 根据输入张量形状计算总元素个数
    //    TilingContext::GetInputShape 返回的是 StorageShape*，
    //    StorageShape 没有 GetDim/GetDimNum，需要先 GetStorageShape() 拿到 Shape 再调用
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    const gert::Shape shape = inputShape->GetStorageShape();
    uint32_t totalLength = 1;
    for (int i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= shape.GetDim(i);
    }

    // 2. 设置核间分块数(BlockDim)与 TilingData
    context->SetBlockDim(BLOCK_DIM);
    tilingData.totalLength = totalLength;
    tilingData.tileNum = TILE_NUM;

    // 3. 将 tiling 数据写入 RawTilingData（gert::TilingData 仅提供 GetData，需 memcpy）
    auto rawTilingData = context->GetRawTilingData();
    memcpy(rawTilingData->GetData(), &tilingData, sizeof(tilingData));

    // 4. Tanh 计算不需要额外 workspace
    //    GetWorkspaceSizes 需要传入 workspace 个数，返回 size_t*
    context->GetWorkspaceSizes(1)[0] = 0;

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
