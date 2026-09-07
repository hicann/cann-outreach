// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype_x = tensor_x->GetDataType();
    const ge::DataType dtype_y = tensor_y->GetDataType();
    if (dtype_x != dtype_y) {
        return ge::GRAPH_FAILED;
    }
    if (dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t length_x = tensor_x->GetShapeSize();
    const uint64_t length_y = tensor_y->GetShapeSize();
    if (length_x == 0 || length_x != length_y) {
        return ge::GRAPH_FAILED;
    }
    if (length_x > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    // 为保证每个核处理相同数量元素，同时尽量使用更多核，选择
    // 不超过AIV核数且能整除总元素数的最大blockDim。
    uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
    if (block_dim == 0) {
        block_dim = 1;
    }
    if (block_dim > length_x) {
        block_dim = static_cast<uint32_t>(length_x);
    }
    while (block_dim > 1 && (length_x % block_dim) != 0) {
        --block_dim;
    }

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->length = static_cast<uint32_t>(length_x);

    context->SetBlockDim(block_dim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    const uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

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
    const ge::DataType input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
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
