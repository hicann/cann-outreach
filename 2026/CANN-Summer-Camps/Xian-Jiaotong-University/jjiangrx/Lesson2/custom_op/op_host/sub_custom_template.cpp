#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
    const gert::StorageShape *xShape = context->GetInputShape(0);
    tiling->totalLength = static_cast<uint32_t>(xShape->GetStorageShape().GetShapeSize());
    tiling->tileNum = TILE_NUM;

    context->SetBlockDim(BLOCK_DIM);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
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
}
