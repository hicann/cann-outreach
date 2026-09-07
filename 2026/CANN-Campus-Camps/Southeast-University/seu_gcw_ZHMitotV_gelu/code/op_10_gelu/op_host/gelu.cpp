// Host侧 Tiling 实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // ============================================================
    // 1. 获取平台信息
    // ============================================================
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    const uint32_t coreNum =
        static_cast<uint32_t>(
            platform.GetCoreNumAiv());

    uint64_t ubSize = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    // ============================================================
    // 2. 获取输入 Tensor
    // ============================================================
    const gert::Tensor *inputTensor =
        context->GetRequiredInputTensor(0);

    const ge::DataType dtype =
        inputTensor->GetDataType();

    const uint32_t dtypeSize =
        static_cast<uint32_t>(
            ge::GetSizeByDataType(dtype));

    const uint32_t totalLength =
        static_cast<uint32_t>(
            inputTensor->GetShapeSize());

    // ============================================================
    // 3. 设置 Tiling Key
    // ============================================================
    const uint32_t DT_INPUT_X =
        static_cast<uint32_t>(dtype);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    // ============================================================
    // 4. 32 Byte 对齐粒度
    //
    // float16:
    //     32 / 2 = 16 elements
    //
    // float32:
    //     32 / 4 = 8 elements
    // ============================================================
    const uint32_t alignElements =
        32 / dtypeSize;

    // ============================================================
    // 5. 计算 Tile 长度
    //
    // 一个 Tile 至少需要：
    //
    // input
    // output
    // 中间计算
    //
    // 因此这里保守使用 UB / 4。
    // ============================================================
    uint32_t tileLength =
        static_cast<uint32_t>(
            (ubSize / 4) / dtypeSize);

    // 32B 对齐
    tileLength =
        (tileLength / alignElements) *
        alignElements;

    // 最大 Tile
    if (tileLength > 8192) {
        tileLength = 8192;
    }

    // 极端情况下保证至少一个 32B block
    if (tileLength < alignElements) {
        tileLength = alignElements;
    }

    // ============================================================
    // 6. 计算工作块数量
    //
    // 一个工作块 = 32B
    // ============================================================
    const uint32_t workUnits =
        (totalLength + alignElements - 1) /
        alignElements;

    // ============================================================
    // 7. 决定 Core 数
    // ============================================================
    uint32_t blockDim = coreNum;

    if (blockDim > workUnits) {
        blockDim = workUnits;
    }

    if (blockDim == 0) {
        blockDim = 1;
    }

    context->SetBlockDim(blockDim);

    // ============================================================
    // 8. 写入 TilingData
    // ============================================================
    GeluTilingData *tiling =
        context->GetTilingData<GeluTilingData>();

    tiling->length = totalLength;
    tiling->tileLength = tileLength;

    // ============================================================
    // 9. 当前版本不需要额外 Workspace
    // ============================================================
    size_t *workspace =
        context->GetWorkspaceSizes(1);

    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


// ============================================================================
// Shape / DataType 推导
// ============================================================================

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const auto inputShape =
        context->GetInputShape(0);

    auto outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType);

    return GRAPH_SUCCESS;
}

} // namespace ge


// ============================================================================
// Operator Definition
// ============================================================================

namespace ops {

class Gelu : public OpDef {
public:

    explicit Gelu(const char *name)
        : OpDef(name)
    {
        // ------------------------------------------------------------
        // Input
        // ------------------------------------------------------------
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        // ------------------------------------------------------------
        // Output
        // ------------------------------------------------------------
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        // ------------------------------------------------------------
        // Shape / DataType inference
        // ------------------------------------------------------------
        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        // ------------------------------------------------------------
        // Ascend 910B
        // ------------------------------------------------------------
        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

} // namespace ops