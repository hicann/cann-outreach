#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();

  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);

  if (tiling == nullptr || x1_shape == nullptr || x2_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& x_shape = x1_shape->GetStorageShape();
  const gert::Shape& y_shape = x2_shape->GetStorageShape();

  if (x_shape.GetDimNum() != y_shape.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }

  int64_t data_sz = 1;
  for (int i = 0; i < x_shape.GetDimNum(); i++) {
    if (x_shape.GetDim(i) != y_shape.GetDim(i)) {
      return ge::GRAPH_FAILED;
    }
    data_sz *= x_shape.GetDim(i);
  }

  if (data_sz > UINT32_MAX) {
    return ge::GRAPH_FAILED;
  }

  tiling->size = static_cast<uint32_t>(data_sz);

  context->SetBlockDim(8);

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  if (currentWorkspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* y_shape = context->GetInputShape(1);
    gert::Shape* z_shape = context->GetOutputShape(0);

    if (x_shape == nullptr || y_shape == nullptr || z_shape == nullptr) {
        return GRAPH_FAILED;
    }

    if (x_shape->GetDimNum() != y_shape->GetDimNum()) {
        return GRAPH_FAILED;
    }

    for (int i = 0; i < x_shape->GetDimNum(); i++) {
        if (x_shape->GetDim(i) != y_shape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }

    *z_shape = *x_shape;
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