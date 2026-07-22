#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstring>


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TanhCustomTilingData tiling;

    // 总数据量：取输入shape的元素总数
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 使用BLOCK_DIM个核并行计算
    context->SetBlockDim(BLOCK_DIM);

    tiling.totalLength = totalLength;
    tiling.tileNum = TILE_NUM;

    context->SetTilingKey(0);

    // 本算子未使用额外workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    // 将tiling结构体写入原始tiling数据区
    memcpy(context->GetRawTilingData()->GetData(), &tiling, sizeof(TanhCustomTilingData));
    context->GetRawTilingData()->SetDataSize(sizeof(TanhCustomTilingData));

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