
#include "../op_kernel/div_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    // 1. 获取输入总长度（x/y/z 形状相同，8*2048 = 16384）
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    tiling.set_totalLength(totalLength);

    // 2. 切分参数：8 核并行，每核 2048 元素，tile 长度 1024（满足 32B 对齐）
    constexpr uint32_t BLOCK_DIM = 8;
    constexpr uint32_t TILE_LENGTH = 1024;
    tiling.set_tileLength(TILE_LENGTH);

    // 3. 记录数据类型，供 kernel 侧选择模板实例（0=float16, 1=float32）
    auto dtype = context->GetInputDesc(0)->GetDataType();
    tiling.set_dataType(dtype == ge::DT_FLOAT ? 1 : 0);

    // 4. 设置使用的 AI Core 数量
    context->SetBlockDim(BLOCK_DIM);

    // 5. 序列化 tiling 数据，供 kernel 侧 GET_TILING_DATA 读取
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 6. 本算子无需额外 workspace
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0;

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
class DivCustomTemplate : public OpDef {
public:
    explicit DivCustomTemplate(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DivCustomTemplate);
}
