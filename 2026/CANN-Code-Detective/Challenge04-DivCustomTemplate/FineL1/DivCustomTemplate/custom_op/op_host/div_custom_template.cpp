#include "div_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取输入 x 的 Shape
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 计算输入 Tensor 的总元素数量
    uint32_t dataSize = 1;
    const gert::Shape& storageShape = xShape->GetStorageShape();

    for (int32_t i = 0; i < storageShape.GetDimNum(); ++i) {
        dataSize *= static_cast<uint32_t>(storageShape.GetDim(i));
    }

    // 构造 TilingData
    DivCustomTemplateTilingData tiling;
    tiling.set_size(dataSize);

    // 使用 8 个 AI Vector Core
    context->SetBlockDim(8);

    // 本算子不需要额外 Workspace
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    if (workspaceSize != nullptr) {
        workspaceSize[0] = 0;
    }

    // 将 TilingData 序列化并传给 Kernel
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(
        tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


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
