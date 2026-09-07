// Host-side tiling and operator definition.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
namespace {
constexpr uint32_t kElementsPerCore = 256;
constexpr uint32_t kMinAlignmentElements = 64;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto *input = context->GetRequiredInputTensor(0);
    if (input == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t core_count = platform.GetCoreNumAiv();
    if (core_count <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t element_count = input->GetShapeSize();
    if (element_count == 0 || element_count > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = input->GetDataType();
    const uint32_t dtype_key = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtype_key);

    auto *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = static_cast<uint32_t>(element_count);

    // The kernel partitions work in 64-element blocks. Keep a small tensor on
    // one core, then scale cores with tensor size to avoid launch overhead.
    const uint32_t aligned_blocks =
        (tiling->length + kMinAlignmentElements - 1) / kMinAlignmentElements;
    const uint32_t requested_cores =
        (tiling->length + kElementsPerCore - 1) / kElementsPerCore;
    const uint32_t used_cores = static_cast<uint32_t>(
        std::min<int64_t>(core_count, std::max<uint32_t>(1, std::min(requested_cores, aligned_blocks))));

    context->SetBlockDim(used_cores);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
