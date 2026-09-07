#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // =========================================================
    // 1. 获取平台信息
    // =========================================================
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    uint32_t numCores =
        platform.GetCoreNumAiv();

    if (numCores == 0) {
        return ge::GRAPH_FAILED;
    }

    // =========================================================
    // 2. 获取输入 Tensor
    // =========================================================
    const gert::Tensor *tensorInputX =
        context->GetRequiredInputTensor(0);

    if (tensorInputX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeInputX =
        tensorInputX->GetDataType();

    uint32_t lengthInputX =
        static_cast<uint32_t>(
            tensorInputX->GetShapeSize());

    uint32_t dtypeSize =
        static_cast<uint32_t>(
            ge::GetSizeByDataType(dtypeInputX));

    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }

    // =========================================================
    // 3. 设置 Tiling Key
    // =========================================================
    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(
            dtypeInputX);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    // =========================================================
    // 4. 32 Byte 对齐元素数
    //
    // FP16 -> 16
    // FP32 -> 8
    // =========================================================
    uint32_t alignElems =
        32 / dtypeSize;

    // =========================================================
    // 5. Tile Length
    //
    // 保持已经验证过正确性的 512。
    // =========================================================
    constexpr uint32_t BASE_TILE_LENGTH = 512;

    uint32_t tileLength =
        (BASE_TILE_LENGTH / alignElems)
        * alignElems;

    if (tileLength < alignElems) {
        tileLength = alignElems;
    }

    // =========================================================
    // 6. 获取 TilingData
    // =========================================================
    GeluTilingData *tiling =
        context->GetTilingData<GeluTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length =
        lengthInputX;

    tiling->tileLength =
        tileLength;

    // =========================================================
    // 7. 完整 32 Byte 区域
    // =========================================================
    uint32_t alignedLength =
        (lengthInputX / alignElems)
        * alignElems;

    // =========================================================
    // 8. Tile 数量
    // =========================================================
    uint32_t tileCount = 0;

    if (alignedLength > 0) {
        tileCount =
            (alignedLength +
             tileLength -
             1)
            / tileLength;
    }

    // =========================================================
    // 9. 根据 Tile 数量选择 Core
    //
    // 大数据使用多个 AI Core，
    // 小数据避免启动过多 Core。
    // =========================================================
    uint32_t activeCores = 1;

    if (tileCount > 0) {

        activeCores =
            tileCount;

        if (activeCores > numCores) {
            activeCores = numCores;
        }

        if (activeCores == 0) {
            activeCores = 1;
        }
    }

    context->SetBlockDim(
        activeCores);

    // =========================================================
    // 10. 不使用 Global Workspace
    // =========================================================
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

// =============================================================
// InferShape
// =============================================================
static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outputShape =
        *inputShape;

    return GRAPH_SUCCESS;
}


// =============================================================
// InferDataType
// =============================================================
static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    ge::DataType inputDataType =
        context->GetInputDataType(0);

    return context->SetOutputDataType(
        0,
        inputDataType);
}

}  // namespace ge


namespace ops {

class Gelu : public OpDef {

public:

    explicit Gelu(const char *name)
        : OpDef(name)
    {
        // =====================================================
        // Input
        // =====================================================
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

        // =====================================================
        // Output
        // =====================================================
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

        // =====================================================
        // Shape / DataType
        // =====================================================
        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        // =====================================================
        // AI Core
        // =====================================================
        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops