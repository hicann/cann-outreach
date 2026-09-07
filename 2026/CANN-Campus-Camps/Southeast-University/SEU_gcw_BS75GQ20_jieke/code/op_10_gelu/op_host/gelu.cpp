#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

// Ascend数据搬运基本对齐字节数。
constexpr uint32_t ALIGN_BYTES = 32;

/*
 * 继续使用当前已经取得较好成绩的分核策略：
 * 每约512字节输入数据分配一个AI Vector Core。
 */
constexpr uint32_t MIN_SPLIT_BYTES = 512;

/*
 * 当前多项式Kernel没有Erf临时空间。
 * 910B的UB容量允许将Tile上限设置为8192。
 */
constexpr uint32_t MAX_TILE_ELEMENTS = 8192;

/*
 * 为TPipe管理信息及其他硬件资源预留空间。
 */
constexpr uint64_t UB_RESERVE_BYTES = 8192;

/*
 * 当前Kernel在多Tile路径下的最坏UB占用：
 *
 * FP32：
 * 输入Queue双缓冲：2 × 4 = 8字节/元素
 * 输出Queue双缓冲：2 × 4 = 8字节/元素
 * squareBuffer：        4字节/元素
 * 合计：               20字节/元素
 *
 * FP16：
 * 输入Queue双缓冲：2 × 2 = 4字节/元素
 * 输出Queue双缓冲：2 × 2 = 4字节/元素
 * squareBuffer：        4字节/元素
 * inputFloatBuffer：    4字节/元素
 * resultFloatBuffer：   4字节/元素
 * 合计：               20字节/元素
 */
constexpr uint32_t MULTI_TILE_BYTES_PER_ELEMENT = 20;

static inline uint64_t CeilDiv(
    uint64_t value,
    uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

static inline uint64_t CeilAlign(
    uint64_t value,
    uint64_t alignment)
{
    return CeilDiv(value, alignment) *
           alignment;
}

static inline uint64_t FloorAlign(
    uint64_t value,
    uint64_t alignment)
{
    return value / alignment *
           alignment;
}

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // ============================================================
    // 1. 获取硬件信息
    // ============================================================
    auto *platformInfo =
        context->GetPlatformInfo();

    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(
            platformInfo);

    const int32_t availableCoreNum =
        platform.GetCoreNumAiv();

    uint64_t ubSize = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    if (availableCoreNum <= 0 ||
        ubSize <= UB_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }

    // ============================================================
    // 2. 获取输入张量信息
    // ============================================================
    const gert::Tensor *inputTensor =
        context->GetRequiredInputTensor(0);

    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputType =
        inputTensor->GetDataType();

    uint32_t typeSize = 0;

    if (inputType == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (inputType == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize =
        inputTensor->GetShapeSize();

    /*
     * 题目规定N至少为1，因此不需要支持空张量。
     * TilingData使用uint32_t，所以检查长度上限。
     */
    if (shapeSize <= 0 ||
        static_cast<uint64_t>(shapeSize) >
            0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength =
        static_cast<uint64_t>(shapeSize);

    /*
     * FP32每32字节包含8个元素；
     * FP16每32字节包含16个元素。
     */
    const uint32_t alignElements =
        ALIGN_BYTES / typeSize;

    // ============================================================
    // 3. 计算启动核数
    // ============================================================
    const uint64_t totalBytes =
        totalLength * typeSize;

    uint64_t targetCoreNum =
        CeilDiv(
            totalBytes,
            MIN_SPLIT_BYTES);

    if (targetCoreNum < 1) {
        targetCoreNum = 1;
    }

    if (targetCoreNum >
        static_cast<uint64_t>(
            availableCoreNum)) {
        targetCoreNum =
            static_cast<uint64_t>(
                availableCoreNum);
    }

    /*
     * 初步计算每核平均元素数。
     */
    const uint64_t averageLength =
        CeilDiv(
            totalLength,
            targetCoreNum);

    /*
     * 每核起始地址必须保持32字节对齐。
     */
    const uint64_t alignedBlockLength =
        CeilAlign(
            averageLength,
            alignElements);

    if (alignedBlockLength == 0 ||
        alignedBlockLength >
            0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockLength =
        static_cast<uint32_t>(
            alignedBlockLength);

    /*
     * 对齐后的blockLength可能略微增大，
     * 重新计算实际使用核数，避免启动空核。
     */
    const uint64_t actualCoreNum =
        CeilDiv(
            totalLength,
            blockLength);

    if (actualCoreNum == 0 ||
        actualCoreNum >
            static_cast<uint64_t>(
                availableCoreNum)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum =
        static_cast<uint32_t>(
            actualCoreNum);

    // ============================================================
    // 4. 根据Kernel实际UB占用计算Tile
    // ============================================================
    const uint64_t availableUb =
        ubSize - UB_RESERVE_BYTES;

    uint64_t maxTileByUb =
        availableUb /
        MULTI_TILE_BYTES_PER_ELEMENT;

    maxTileByUb =
        FloorAlign(
            maxTileByUb,
            alignElements);

    if (maxTileByUb < alignElements) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tileLength =
        blockLength;

    if (tileLength >
        MAX_TILE_ELEMENTS) {
        tileLength =
            MAX_TILE_ELEMENTS;
    }

    if (tileLength >
        maxTileByUb) {
        tileLength =
            maxTileByUb;
    }

    tileLength =
        FloorAlign(
            tileLength,
            alignElements);

    if (tileLength < alignElements) {
        tileLength = alignElements;
    }

    if (tileLength >
        0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    // ============================================================
    // 5. 写入TilingData
    // ============================================================
    GeluTilingData *tiling =
        context->GetTilingData<
            GeluTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength =
        static_cast<uint32_t>(
            totalLength);

    tiling->blockLength =
        blockLength;

    tiling->tileLength =
        static_cast<uint32_t>(
            tileLength);

    // ============================================================
    // 6. 选择模板数据类型
    // ============================================================
    const uint32_t DT_INPUT_X =
        static_cast<uint32_t>(
            inputType);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    context->SetBlockDim(
        usedCoreNum);

    // 本算子不需要额外Workspace。
    size_t *workspace =
        context->GetWorkspaceSizes(1);

    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    /*
     * GELU是逐元素算子，
     * 输出Shape和输入Shape完全相同。
     */
    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    /*
     * 输出类型和输入类型完全相同。
     */
    context->SetOutputDataType(
        0,
        context->GetInputDataType(0));

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char *name)
        : OpDef(name)
    {
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

        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops