#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

#include <cstdint>
#include <limits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
  const gert::StorageShape* xShape = context->GetInputShape(0);
  const gert::StorageShape* yShape = context->GetInputShape(1);
  if (tiling == nullptr || xShape == nullptr || yShape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& xStorageShape = xShape->GetStorageShape();
  const gert::Shape& yStorageShape = yShape->GetStorageShape();
  if (xStorageShape.GetDimNum() != yStorageShape.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }

  uint64_t dataSize = 1;
  constexpr uint64_t maxTilingSize = std::numeric_limits<uint32_t>::max();
  for (int64_t i = 0; i < xStorageShape.GetDimNum(); ++i) {
    const int64_t xDim = xStorageShape.GetDim(i);
    const int64_t yDim = yStorageShape.GetDim(i);
    if (xDim <= 0 || xDim != yDim) {
      return ge::GRAPH_FAILED;
    }
    const uint64_t dim = static_cast<uint64_t>(xDim);
    if (dataSize > maxTilingSize / dim) {
      return ge::GRAPH_FAILED;
    }
    dataSize *= dim;
  }

  tiling->size = static_cast<uint32_t>(dataSize);
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
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* yShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || outputShape == nullptr ||
        xShape->GetDimNum() != yShape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (int64_t i = 0; i < xShape->GetDimNum(); ++i) {
        if (xShape->GetDim(i) != yShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *outputShape = *xShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto xDataType = context->GetInputDataType(0);
    const auto yDataType = context->GetInputDataType(1);
    if (xDataType != yDataType) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, xDataType);
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
