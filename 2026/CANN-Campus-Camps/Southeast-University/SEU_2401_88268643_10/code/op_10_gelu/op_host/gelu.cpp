#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

constexpr uint64_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t MAX_TILE_ELEMENTS = 4096;

static inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::Tensor* inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputType = inputTensor->GetDataType();
    if (inputType != ge::DT_FLOAT && inputType != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(inputType);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    const int64_t shapeSize = inputTensor->GetShapeSize();
    if (shapeSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalLength = static_cast<uint64_t>(shapeSize);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (availableCores == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t typeSize = inputType == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t alignElements = DATA_BLOCK_BYTES / typeSize;
    const uint64_t alignedBlocks = CeilDiv(totalLength, alignElements);
    const uint64_t usedCores =
        alignedBlocks < availableCores ? alignedBlocks : availableCores;

    // Divide aligned 32-byte blocks rather than raw elements.  This uses every
    // useful vector core and keeps each core's GM start address aligned.
    const uint64_t blocksPerCore = alignedBlocks / usedCores;
    const uint64_t remainderBlocks = alignedBlocks % usedCores;
    const uint64_t smallBlockLength = blocksPerCore * alignElements;
    const uint64_t largeBlockLength = (blocksPerCore + 1) * alignElements;
    const uint64_t maxBlockLength =
        remainderBlocks == 0 ? smallBlockLength : largeBlockLength;
    const uint32_t tileLength = static_cast<uint32_t>(
        maxBlockLength < MAX_TILE_ELEMENTS ? maxBlockLength : MAX_TILE_ELEMENTS);

    GeluTilingData* tilingData = context->GetTilingData<GeluTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->totalLength = totalLength;
    tilingData->largeBlockLength = largeBlockLength;
    tilingData->smallBlockLength = smallBlockLength;
    tilingData->largeCoreCount = static_cast<uint32_t>(remainderBlocks);
    tilingData->tileLength = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType inputType = context->GetInputDataType(0);
    if (inputType != ge::DT_FLOAT && inputType != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputType);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name)
    {
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

} // namespace ops
