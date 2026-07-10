
#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>
#include <limits>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  if (x1_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  uint64_t dataSize = 1;
  const auto &storageShape = x1_shape->GetStorageShape();
  for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
    const int64_t dim = storageShape.GetDim(i);
    if (dim <= 0 || dataSize > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      return ge::GRAPH_FAILED;
    }
    dataSize *= static_cast<uint64_t>(dim);
  }
  if (dataSize == 0) {
    return ge::GRAPH_FAILED;
  }
  tiling->size = dataSize;
  context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(8, dataSize)));
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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(SubCustomTemplate);
}
