// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
namespace {
constexpr uint64_t DATA_BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
constexpr uint32_t MAX_TILE_ELEMENTS = 4096;
constexpr uint64_t VECTOR_ELEMENTS = 64;  // One full fp32 vector repeat.
// Initial tuning value: avoid launching one core per 32 bytes for tiny inputs.
// Benchmark 128 / 256 / 512 on the target device to tune small-tensor latency.
constexpr uint64_t TARGET_ELEMENTS_PER_CORE = 256;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputDtype = inputTensor->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeBytes = static_cast<uint32_t>(ge::GetSizeByDataType(inputDtype));
    const uint64_t alignElements = DATA_BLOCK_BYTES / dtypeBytes;
    const uint64_t totalLength = static_cast<uint64_t>(inputTensor->GetShapeSize());

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t maxAivCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (maxAivCores == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // Budget for two queue slots even if a one-tile core later uses one slot.
    // No high-level Erfc scratch is needed. The reserve also covers the
    // rational kernel's six 32-byte coefficient blocks (192 bytes).
    const uint64_t workTensorCount = inputDtype == ge::DT_FLOAT16 ? 5 : 4;
    const uint64_t bytesPerElement = 4 * dtypeBytes + workTensorCount * sizeof(float);
    if (ubSize < UB_RESERVED_BYTES + bytesPerElement * VECTOR_ELEMENTS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tileLength = (ubSize - UB_RESERVED_BYTES) / bytesPerElement;
    tileLength = std::min<uint64_t>(tileLength, MAX_TILE_ELEMENTS);
    tileLength = (tileLength / VECTOR_ELEMENTS) * VECTOR_ELEMENTS;

    // Every core begins at a 32-byte-aligned address.  Consequently only the
    // final core can own a non-aligned tail.
    const uint64_t wantedCores = std::max<uint64_t>(1, CeilDiv(totalLength, TARGET_ELEMENTS_PER_CORE));
    const uint32_t requestedCores = static_cast<uint32_t>(
        std::min<uint64_t>(maxAivCores, wantedCores));
    const uint64_t blockLength = AlignUp(CeilDiv(std::max<uint64_t>(totalLength, 1), requestedCores),
                                               alignElements);
    const uint32_t usedCores = static_cast<uint32_t>(
        std::max<uint64_t>(1, CeilDiv(totalLength, blockLength)));
    // Allocate only the useful tile size, with a full fp32 repeat as the minimum.
    tileLength = std::min<uint64_t>(tileLength, AlignUp(blockLength, VECTOR_ELEMENTS));

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->reserved = 0;

    context->SetBlockDim(usedCores);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
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
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}  // namespace ops
