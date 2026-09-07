#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static constexpr uint32_t BLOCK_BYTES = 32U;
static constexpr uint32_t MIN_ELEMENTS_PER_CORE = 256U;
static constexpr uint64_t UB_RESERVED_BYTES = 4096U;
static constexpr uint64_t MAX_TILE_BYTES = 65504U;

static inline uint64_t AlignDown(
    const uint64_t value,
    const uint64_t alignment)
{
    return value / alignment * alignment;
}

static inline uint64_t AlignUp(
    const uint64_t value,
    const uint64_t alignment)
{
    return (value + alignment - 1U) /
           alignment * alignment;
}

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    const uint32_t platformCoreNum =
        static_cast<uint32_t>(
            platform.GetCoreNumAiv());

    uint64_t ubSize = 0U;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    const gert::Tensor *inputTensor =
        context->GetRequiredInputTensor(0);

    const ge::DataType inputDtype =
        inputTensor->GetDataType();

    const uint32_t dtypeSize =
        static_cast<uint32_t>(
            ge::GetSizeByDataType(inputDtype));

    const uint32_t length =
        static_cast<uint32_t>(
            inputTensor->GetShapeSize());

    const uint32_t elementsPerBlock =
        BLOCK_BYTES / dtypeSize;

    const uint32_t fullBlockCount =
        length / elementsPerBlock;

    const uint32_t tailElements =
        length % elementsPerBlock;

    /*
     * 恢复原来的256元素/核策略。
     * 不再使用上一版激进增加核数的方案。
     */
    uint32_t coreNum = 1U;

    if (fullBlockCount != 0U) {
        const uint32_t wantedCoreNum =
            (length + MIN_ELEMENTS_PER_CORE - 1U) /
            MIN_ELEMENTS_PER_CORE;

        coreNum = std::min(
            std::max(platformCoreNum, 1U),
            wantedCoreNum);

        // 保证每个核至少拥有一个完整32字节块
        coreNum = std::min(
            coreNum,
            fullBlockCount);

        coreNum = std::max(coreNum, 1U);
    }

    const uint32_t baseBlockCount =
        fullBlockCount / coreNum;

    const uint32_t extraBlockCores =
        fullBlockCount % coreNum;

    const uint32_t largestRegularBlockCount =
        baseBlockCount +
        ((extraBlockCores != 0U) ? 1U : 0U);

    const uint32_t lastCoreBlockCount =
        baseBlockCount +
        (((coreNum - 1U) < extraBlockCores)
             ? 1U
             : 0U);

    const uint32_t largestRegularLength =
        largestRegularBlockCount *
        elementsPerBlock;

    const uint32_t lastCoreLength =
        lastCoreBlockCount *
            elementsPerBlock +
        tailElements;

    uint32_t maxCoreLength =
        std::max(
            largestRegularLength,
            lastCoreLength);

    maxCoreLength = std::max(
        maxCoreLength,
        elementsPerBlock);

    const uint64_t alignedMaxCoreBytes =
        AlignUp(
            static_cast<uint64_t>(
                maxCoreLength) *
                dtypeSize,
            BLOCK_BYTES);

    const uint64_t usableUbSize =
        (ubSize > UB_RESERVED_BYTES)
            ? ubSize - UB_RESERVED_BYTES
            : ubSize;

    uint32_t tmpBufferSize = 0U;
    uint64_t tileBytes = BLOCK_BYTES;

    if (inputDtype == ge::DT_FLOAT) {
        /*
         * float32采用Kernel中的快速近似路径，
         * 需要input、output、scaled三个Buffer。
         */
        tileBytes = usableUbSize / 3U;

        tileBytes = std::min(
            tileBytes,
            MAX_TILE_BYTES);

        tileBytes = std::min(
            tileBytes,
            alignedMaxCoreBytes);

        tileBytes = AlignDown(
            tileBytes,
            BLOCK_BYTES);

        tileBytes = std::max<uint64_t>(
            tileBytes,
            BLOCK_BYTES);
    } else {
        /*
         * float16继续使用Erf路径，需要Erf临时空间。
         */
        uint32_t maxLiveNodeCount = 0U;
        uint32_t extraBuffer = 0U;

        AscendC::GetErfTmpBufferFactorSize(
            dtypeSize,
            maxLiveNodeCount,
            extraBuffer);

        const uint64_t tensorFactor =
            3U +
            static_cast<uint64_t>(
                maxLiveNodeCount);

        const uint64_t usableForTensor =
            (usableUbSize > extraBuffer)
                ? usableUbSize - extraBuffer
                : BLOCK_BYTES * tensorFactor;

        tileBytes =
            usableForTensor / tensorFactor;

        tileBytes = std::min(
            tileBytes,
            MAX_TILE_BYTES);

        tileBytes = std::min(
            tileBytes,
            alignedMaxCoreBytes);

        tileBytes = AlignDown(
            tileBytes,
            BLOCK_BYTES);

        tileBytes = std::max<uint64_t>(
            tileBytes,
            BLOCK_BYTES);

        const uint64_t tmpSize =
            tileBytes *
                static_cast<uint64_t>(
                    maxLiveNodeCount) +
            extraBuffer;

        tmpBufferSize =
            static_cast<uint32_t>(
                AlignUp(
                    tmpSize,
                    BLOCK_BYTES));
    }

    const uint32_t tileLength =
        static_cast<uint32_t>(
            tileBytes / dtypeSize);

    GeluTilingData *tiling =
        context->GetTilingData<
            GeluTilingData>();

    tiling->coreNum =
        coreNum;

    tiling->baseBlockCount =
        baseBlockCount;

    tiling->extraBlockCores =
        extraBlockCores;

    tiling->tailElements =
        tailElements;

    tiling->tileLength =
        tileLength;

    tiling->tmpBufferSize =
        tmpBufferSize;

    const uint32_t DT_INPUT_X =
        static_cast<uint32_t>(
            inputDtype);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    context->SetBlockDim(coreNum);

    size_t *workspaceSizes =
        context->GetWorkspaceSizes(1);

    workspaceSizes[0] = 0U;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const ge::DataType inputDtype =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDtype);

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
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops