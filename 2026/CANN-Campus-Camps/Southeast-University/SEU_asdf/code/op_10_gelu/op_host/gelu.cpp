#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCores = platform.GetCoreNumAiv();
    if (numCores <= 0) {
        numCores = 1;
    }

    const gert::Tensor *input = context->GetRequiredInputTensor(0);
    if (input == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = input->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    const uint32_t length = static_cast<uint32_t>(input->GetShapeSize());
    if (length == 0U) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t coreLimit = static_cast<uint32_t>(numCores);
    if (coreLimit > length) {
        coreLimit = length;
    }
    if (coreLimit == 0U) {
        coreLimit = 1U;
    }

    const uint32_t avgPerCore = (length + coreLimit - 1U) / coreLimit;
    const uint32_t alignElems = 32U / dtypeSize;

    uint32_t blockLength;
    if (length <= 8192U) {
        blockLength = avgPerCore;
    } else {
        blockLength = ((avgPerCore + alignElems - 1U) / alignElems) * alignElems;
    }
    if (blockLength == 0U) {
        blockLength = 1U;
    }

    uint32_t usedCores = (length + blockLength - 1U) / blockLength;
    if (usedCores == 0U) {
        usedCores = 1U;
    }
    if (usedCores > coreLimit) {
        usedCores = coreLimit;
    }

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->blockLength = blockLength;

    context->SetBlockDim(usedCores);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops
