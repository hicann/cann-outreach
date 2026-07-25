#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t MAX_BLOCK_DIM = 40;
constexpr uint32_t TILE_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTemplateTilingData* tiling =
        context->GetTilingData<AddCustomTemplateTilingData>();

    const int64_t shapeSize =
        context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    if (shapeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = static_cast<uint32_t>(shapeSize);
    const uint32_t blockDim =
        totalLength < MAX_BLOCK_DIM ? totalLength : MAX_BLOCK_DIM;

    context->SetBlockDim(blockDim);
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *x1Shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore()
            .AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);
}  