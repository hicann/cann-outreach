
#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::StorageShape *x1_shape = context->GetInputShape(0);
  if (x1_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t totalLength = x1_shape->GetOriginShape().GetShapeSize();
  const uint32_t blockDim = totalLength == 0 ? 1 : (totalLength < 40 ? totalLength : 40);
  context->SetBlockDim(blockDim);
  tiling->totalLength = totalLength;
  tiling->tileNum = 2;
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* y1_shape = context->GetInputShape(1);
    gert::Shape* z_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y1_shape == nullptr || z_shape == nullptr) {
        return GRAPH_FAILED;
    }
    if (*x1_shape != *y1_shape) {
        return GRAPH_FAILED;
    }
    *z_shape = *x1_shape;
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
