// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取输入 Tensor
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);

    // 获取输入数据类型，并通过 TilingKey 传递给 Kernel 模板
    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取并填写 Tiling 数据
    MulTilingData *tiling = context->GetTilingData<MulTilingData>();

    // x 的总元素个数，题目中为 8 * 2048 = 16384
    tiling->length = tensor_x->GetShapeSize();

    // 每个 Core 内进一步划分为 8 个 tile
    tiling->tileNum = TILE_NUM;

    // 使用 8 个 AI Core
    context->SetBlockDim(8);

    // 本算子不需要额外 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    // z 的 shape 与 x 完全一致
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // z 的 dtype 与 x 完全一致
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops