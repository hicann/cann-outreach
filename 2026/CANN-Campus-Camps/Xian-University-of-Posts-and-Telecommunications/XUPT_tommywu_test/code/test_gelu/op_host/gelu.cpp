// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
namespace {
constexpr uint32_t TILE_LENGTH = 2048;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv == 0) {
        num_cores_aiv = 1;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t total_length = tensor_x->GetShapeSize();
    uint32_t block_dim = num_cores_aiv;
    if (total_length == 0) {
        block_dim = 1;
    } else if (total_length < block_dim) {
        block_dim = total_length;
    }

    uint32_t block_length = (total_length + block_dim - 1) / block_dim;
    uint32_t tile_num = (block_length + TILE_LENGTH - 1) / TILE_LENGTH;
    if (tile_num == 0) {
        tile_num = 1;
    }

    uint32_t dt_x = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dt_x);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = total_length;
    tiling->tileNum = tile_num;

    context->SetBlockDim(block_dim);

    size_t *current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *input_shape = context->GetInputShape(0);
    gert::Shape *output_shape = context->GetOutputShape(0);
    if (input_shape == nullptr || output_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops