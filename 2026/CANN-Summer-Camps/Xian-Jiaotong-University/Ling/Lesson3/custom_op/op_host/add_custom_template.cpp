#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // Ascend910B 设置BlockDim为40
    context->SetBlockDim(40);
    tiling->totalLength = totalLength;
    // 每核23040元素，单个tile分配2880元素
    tiling->tileNum = 8;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
}
}

namespace ops
{
class AddCustomTemplate : public OpDef
{
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
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);
}