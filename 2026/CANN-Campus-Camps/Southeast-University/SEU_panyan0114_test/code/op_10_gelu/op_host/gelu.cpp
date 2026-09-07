// Host-side definition and tiling for GELU.
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t DEFAULT_TILE_LENGTH = 1024;

template <typename T>
inline T CeilDiv(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
inline T AlignUp(T value, T align)
{
    return CeilDiv(value, align) * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    const ge::DataType inputDtype = inputTensor->GetDataType();
    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(inputDtype));
    const uint64_t totalLength = static_cast<uint64_t>(inputTensor->GetShapeSize());

    // Select the dtype-specialized kernel generated from tiling_key_gelu.h.
    const uint32_t DT_INPUT_X = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileLength = DEFAULT_TILE_LENGTH;

    // Keep every non-last core's GM start address 32-byte aligned.  The last
    // short fragment is copied with DataCopyPad on the kernel side.
    const uint64_t elemPerBlock = BYTE_BLOCK / dtypeSize;
    const uint64_t desiredCoreNum = std::max<uint64_t>(
        1, std::min<uint64_t>(availableCoreNum == 0 ? 1 : availableCoreNum,
                              CeilDiv(totalLength, static_cast<uint64_t>(DEFAULT_TILE_LENGTH))));
    const uint64_t rawBlockLength = CeilDiv(totalLength, desiredCoreNum);
    const uint64_t blockLength = AlignUp(rawBlockLength, elemPerBlock);
    const uint32_t actualCoreNum = static_cast<uint32_t>(CeilDiv(totalLength, blockLength));

    tiling->blockLength = blockLength;
    context->SetBlockDim(std::max<uint32_t>(1, actualCoreNum));

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return GRAPH_SUCCESS;
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops
