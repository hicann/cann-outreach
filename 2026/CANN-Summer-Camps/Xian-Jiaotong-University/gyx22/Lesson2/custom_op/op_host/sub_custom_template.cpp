#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        data_sz *= x1_shape->GetStorageShape().GetDim(i);

    tiling->totalLength = data_sz;
    tiling->tileNum = 4;

    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr) {
        auto dtype = inputDesc->GetDataType();
        tiling->dtype = (dtype == ge::DT_FLOAT) ? 0 : 1;
    } else {
        tiling->dtype = 0;
    }

    context->SetBlockDim(8);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* y_shape_in = context->GetInputShape(1);
    gert::Shape* out_shape = context->GetOutputShape(0);

    if (x1_shape == nullptr || y_shape_in == nullptr || out_shape == nullptr) {
        return GRAPH_FAILED;
    }

    if (*x1_shape != *y_shape_in) {
        return GRAPH_FAILED;
    }

    *out_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto x_dtype = context->GetInputDataType(0);
    const auto y_dtype = context->GetInputDataType(1);
    if (x_dtype != y_dtype) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SubCustomTemplate : public OpDef {
public:
    explicit SubCustomTemplate(const char* name) : OpDef(name)
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

OP_ADD(SubCustomTemplate);
} // namespace ops