// Host-side tiling implementation
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    uint32_t length_x = tensor_x->GetShapeSize();
    tiling->length = length_x;
    context->SetBlockDim(8);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);
    *z_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
