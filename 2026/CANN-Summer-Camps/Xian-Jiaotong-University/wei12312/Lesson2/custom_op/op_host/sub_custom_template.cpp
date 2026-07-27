#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
constexpr uint32_t USE_CORE_NUM = 8;
constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_ALIGNMENT = USE_CORE_NUM * TILE_NUM * BUFFER_NUM;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  uint64_t dataSize = 1;
  for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); ++i) {
    const int64_t dim = x1_shape->GetStorageShape().GetDim(i);
    if (dim <= 0 ||
        dataSize > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(dim)) {
      return ge::GRAPH_FAILED;
    }
    dataSize *= static_cast<uint64_t>(dim);
  }
  if (dataSize % DATA_ALIGNMENT != 0) {
    return ge::GRAPH_FAILED;
  }

  tiling->size = static_cast<uint32_t>(dataSize);
  context->SetBlockDim(USE_CORE_NUM);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
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
