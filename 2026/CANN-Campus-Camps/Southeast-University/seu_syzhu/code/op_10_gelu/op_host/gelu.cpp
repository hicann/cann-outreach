#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr || availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputType = inputTensor->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(inputType);
    if (typeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength =
        static_cast<uint32_t>(inputTensor->GetShapeSize());
    const uint32_t templateType = static_cast<uint32_t>(inputType);
    ASCENDC_TPL_SEL_PARAM(context, templateType);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t alignNum = 32U / static_cast<uint32_t>(typeSize);
    const uint32_t alignedBlocks =
        (totalLength + alignNum - 1U) / alignNum;
    const uint32_t usefulCores = std::min<uint32_t>(
        static_cast<uint32_t>(availableCores), alignedBlocks);
    const uint32_t blockDim = std::max<uint32_t>(usefulCores, 1U);
    const uint32_t blocksPerCore = alignedBlocks / blockDim;

    tiling->length = totalLength;
    tiling->smallBlockLength = blocksPerCore * alignNum;
    tiling->bigCoreCount = alignedBlocks % blockDim;

    context->SetBlockDim(blockDim);

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
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
    explicit Gelu(const char *name) : OpDef(name)
    {
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


