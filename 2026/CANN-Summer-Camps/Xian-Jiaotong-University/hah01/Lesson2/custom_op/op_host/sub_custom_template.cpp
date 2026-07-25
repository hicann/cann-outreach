#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
constexpr uint32_t USED_CORE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tilingData = context->GetTilingData<SubCustomTemplateTilingData>();
    const auto& inputStorageShape = context->GetInputShape(0)->GetStorageShape();

    uint32_t totalLength = 1;
    int64_t axis = 0;
    const int64_t rank = inputStorageShape.GetDimNum();
    while (axis < rank) {
        totalLength *= static_cast<uint32_t>(inputStorageShape.GetDim(axis));
        ++axis;
    }

    tilingData->size = totalLength;
    context->SetBlockDim(USED_CORE_NUM);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class SubCustomTemplate : public OpDef {
public:
    explicit SubCustomTemplate(const char* name) : OpDef(name)
    {
        RegisterInput("x");
        RegisterInput("y");
        RegisterOutput("z");

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }

private:
    void RegisterInput(const char* tensorName)
    {
        this->Input(tensorName)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void RegisterOutput(const char* tensorName)
    {
        this->Output(tensorName)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }
};

OP_ADD(SubCustomTemplate);
}  // namespace ops