/* div_custom_template.cpp (op_host)
 *
 * DivCustomTemplate 算子 host 侧实现：
 *   - TilingFunc：计算输入总元素数、记录输入数据类型、设置核内 tile 切分数
 *     与参与计算的核数（BlockDim = 8，保证整除与 32B 对齐）；
 *   - InferShape / InferDataType：输出 z 与输入 x 的形状、数据类型一致；
 *   - OpDef 注册：输入 x/y、输出 z 均支持 float16 / float32，ND 格式，
 *     注册到 ascend910b 平台。
 */

#include "../op_kernel/div_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
// 参与计算的核数：本期任务数据量为 (8, 2048) = 16384 个元素，取 8 核均分，
// 每个核 2048 个元素，可继续被 tileNum * BUFFER_NUM = 16 整除（每片 128 个元素，
// float16 下 256B / float32 下 512B，均满足 32B 对齐要求）。
// 注：ascend910b1 实际有 48 个向量核，本算子数据量小，按 8 核切分即可保证
// 整除与对齐（与 Ascend C 官方 AddCustom 样例的切分方式一致）。
constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  DivCustomTemplateTilingData *tiling = context->GetTilingData<DivCustomTemplateTilingData>();

  // 1. 计算输入张量的总元素数（各维度连乘）
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  uint32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  }
  tiling->totalLength = data_sz;
  // 2. 核内 tile 切分数（配合双缓冲流水）
  tiling->tileNum = DIV_TILE_NUM;
  // 3. 记录输入数据类型，供 kernel 侧分发核函数模板实例
  const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
  tiling->dtype = (dtype == ge::DT_FLOAT16) ? DIV_DTYPE_FLOAT16 : DIV_DTYPE_FLOAT;
  // 4. 设置参与计算的核数
  context->SetBlockDim(BLOCK_DIM);
  // 5. 本算子不需要额外 workspace
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
