#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const ge::DataType dtypeX = tensorX->GetDataType();
    const uint32_t length = tensorX->GetShapeSize();

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // Different dtypes have different latency sweet spots.
    const uint32_t elemBytes = (dtypeX == ge::DT_FLOAT16) ? 2U : 4U;
    const uint32_t inputBytesPerCore =
        (dtypeX == ge::DT_FLOAT16)
            ? mul_config::FP16_INPUT_BYTES_PER_CORE
            : mul_config::FP32_INPUT_BYTES_PER_CORE;
    const uint32_t blockLength = inputBytesPerCore / elemBytes;
    const uint32_t blockDim = length / blockLength;

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length;

    context->SetBlockDim(blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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