#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);
  if (tiling == nullptr || x1_shape == nullptr || x2_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto& x1_storage_shape = x1_shape->GetStorageShape();
  const auto& x2_storage_shape = x2_shape->GetStorageShape();
  if (x1_storage_shape.GetDimNum() != x2_storage_shape.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }

  uint64_t data_sz = 1;
  for (size_t i = 0; i < x1_storage_shape.GetDimNum(); ++i) {
    const int64_t x1_dim = x1_storage_shape.GetDim(i);
    const int64_t x2_dim = x2_storage_shape.GetDim(i);
    if (x1_dim <= 0 || x1_dim != x2_dim ||
        data_sz > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(x1_dim)) {
      return ge::GRAPH_FAILED;
    }
    data_sz *= static_cast<uint64_t>(x1_dim);
  }

  constexpr uint32_t block_dim = 8;
  constexpr uint32_t tiles_per_block = 16;
  constexpr uint32_t elements_per_data_block = 16;
  if (data_sz % (block_dim * tiles_per_block * elements_per_data_block) != 0) {
    return ge::GRAPH_FAILED;
  }

  tiling->size = static_cast<uint32_t>(data_sz);
  context->SetBlockDim(block_dim);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr ||
        x1_shape->GetDimNum() != x2_shape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < x1_shape->GetDimNum(); ++i) {
        if (x1_shape->GetDim(i) != x2_shape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
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
