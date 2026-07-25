
#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

  // The target input contains 45 * 20480 = 921600 float elements.  Using all
  // 40 vector cores gives 23040 elements per core.  The kernel uses two
  // buffers and tileNum denotes the number of tiles handled by each buffer,
  // therefore tileNum = 2 produces four 5760-element tiles per core.
  // Compared with the original 360 tiny tiles, this removes almost all of the
  // queue and DataCopy setup overhead while keeping every transfer 32-byte
  // aligned.
  context->SetBlockDim(40);
  tiling->totalLength = totalLength;
  tiling->tileNum = 2;
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}
