#include "../op_kernel/div_custom_template_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  DivCustomTemplateTilingData *tiling = context->GetTilingData<DivCustomTemplateTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  // 用 int64_t 累乘 shape 各维度，避免大 tensor 时 int32_t 溢出导致 tiling->size 失真、kernel 越界
  int64_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  // 元素总数超过 uint32_t 可表示范围时直接返回失败，避免 kernel 侧按错误的 size 访问
  if (data_sz < 0 || data_sz > UINT32_MAX) {
    return ge::GRAPH_FAILED;
  }
  tiling->size = static_cast<uint32_t>(data_sz);

  // 记录输入数据类型供 kernel 侧选择模板实例：0 = float16，1 = float32
  const gert::Tensor* inputTensor = context->GetInputTensor(0);
  tiling->dtype = (inputTensor != nullptr && inputTensor->GetDataType() == ge::DT_FLOAT) ? 1 : 0;

  // 核数自适应：每个核最多处理 2048 个元素（与 kernel 侧 TILE_SIZE 保持一致），最多启动 8 个核
  constexpr uint32_t MAX_CORE_NUM = 8;
  constexpr uint32_t TILE_SIZE = 2048;
  uint32_t blockDim = std::min<uint32_t>(MAX_CORE_NUM, (static_cast<uint32_t>(data_sz) + TILE_SIZE - 1) / TILE_SIZE);
  if (blockDim == 0) {
    blockDim = 1;
  }
  context->SetBlockDim(blockDim);
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
class DivCustomTemplate : public OpDef {
public:
    explicit DivCustomTemplate(const char* name) : OpDef(name)
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

OP_ADD(DivCustomTemplate);
}
