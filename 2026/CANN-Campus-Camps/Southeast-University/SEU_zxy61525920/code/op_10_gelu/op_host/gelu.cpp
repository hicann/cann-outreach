// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_input_x =
        context->GetRequiredInputTensor(0);

    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    uint32_t length_input_x =
        static_cast<uint32_t>(tensor_input_x->GetShapeSize());

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData *tiling =
        context->GetTilingData<GeluTilingData>();
    tiling->length = length_input_x;

    // 避免启动无数据可处理的空闲核
    uint32_t block_dim =
        static_cast<uint32_t>(num_cores_aiv > 0 ? num_cores_aiv : 1);

    if (block_dim > length_input_x) {
        block_dim = length_input_x;
    }

    if (block_dim == 0) {
        block_dim = 1;
    }

    context->SetBlockDim(block_dim);

    size_t *current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *input_shape = context->GetInputShape(0);
    gert::Shape *output_shape = context->GetOutputShape(0);

    if (input_shape == nullptr || output_shape == nullptr) {
        return GRAPH_FAILED;
    }

    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(
        0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);
}  // namespace ops