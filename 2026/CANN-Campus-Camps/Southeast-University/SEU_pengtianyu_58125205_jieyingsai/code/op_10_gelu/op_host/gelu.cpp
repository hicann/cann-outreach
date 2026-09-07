#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

constexpr uint32_t ALIGN_BYTES = 32;

// 保持当前分核策略，本轮不调整。
constexpr uint32_t MIN_SPLIT_BYTES = 512;

// 本轮唯一的调优参数：
// 从2048提高到8192。
// 这是上限，实际Tile还会根据UB容量缩小。
constexpr uint32_t MAX_TILE_ELEMENTS = 8192;

// 按双缓冲计算最坏情况下的输入输出空间。
constexpr uint32_t BUFFER_NUM = 2;

// 为其他内部资源保留空间。
constexpr uint64_t UB_RESERVE_BYTES = 8192;

static ge::graphStatus TilingFunc(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // ------------------------------------------------------------
    // 1. 查询硬件资源
    // ------------------------------------------------------------
    auto* platformInfo =
        context->GetPlatformInfo();

    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(platformInfo);

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

    // ------------------------------------------------------------
    // 2. 查询输入类型和总元素数量
    // ------------------------------------------------------------
    const gert::Tensor* inputTensor =
        context->GetRequiredInputTensor(0);

    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype =
        inputTensor->GetDataType();

    uint32_t typeSize = 0;

    if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize =
        inputTensor->GetShapeSize();

    // 当前工程的Tiling字段使用uint32_t。
    // 防止超出范围时静默截断。
    if (shapeSize <= 0 ||
        static_cast<uint64_t>(shapeSize) >
            0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength =
        static_cast<uint64_t>(shapeSize);

    const uint32_t alignElements =
        ALIGN_BYTES / typeSize;

    // ------------------------------------------------------------
    // 3. 保持当前512字节分核策略
    // ------------------------------------------------------------
    uint64_t targetCoreNum =
        (totalLength * typeSize + MIN_SPLIT_BYTES - 1)
        / MIN_SPLIT_BYTES;

    if (targetCoreNum < 1) {
        targetCoreNum = 1;
    }

    if (targetCoreNum >
        static_cast<uint64_t>(availableCoreNum)) {
        targetCoreNum =
            static_cast<uint64_t>(availableCoreNum);
    }

    const uint64_t averageLength =
        (totalLength + targetCoreNum - 1)
        / targetCoreNum;

    // 每核起始地址保持32字节对齐。
    const uint64_t alignedBlockLength =
        ((averageLength + alignElements - 1)
            / alignElements)
        * alignElements;

    if (alignedBlockLength > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockLength =
        static_cast<uint32_t>(alignedBlockLength);

    const uint32_t usedCoreNum =
        static_cast<uint32_t>(
            (totalLength + blockLength - 1)
            / blockLength);

    // ------------------------------------------------------------
    // 4. 初始Tile长度
    //
    // 小输入：仍按每核实际分块长度设置。
    // 大输入：允许尝试更大的Tile，减少循环次数。
    // ------------------------------------------------------------
    uint32_t tileLength =
        blockLength < MAX_TILE_ELEMENTS
            ? blockLength
            : MAX_TILE_ELEMENTS;

    uint32_t erfTmpBytes = 0;

    // 保持上一版保守的空间预算，避免同时改变多个因素。
    //
    // FP16预算：
    // 输入输出双缓冲 + 3块FP32工作空间。
    //
    // FP32预算：
    // 输入输出双缓冲 + 2块FP32工作空间。
    // 当前Kernel实际少用一块，额外空间保留作余量。
    const uint32_t floatBufferNum =
        typeSize == 2 ? 3U : 2U;

    const uint64_t availableUb =
        ubSize - UB_RESERVE_BYTES;

    // ------------------------------------------------------------
    // 5. 查询Erf空间需求，确保整个Tile可以放入UB
    // ------------------------------------------------------------
    while (true) {
        std::vector<int64_t> dims{
            static_cast<int64_t>(tileLength)
        };

        ge::Shape erfShape(dims);

        uint32_t maxTmpBytes = 0;
        uint32_t minTmpBytes = 0;

        // 无论输入是FP16还是FP32，
        // Kernel中的Erf都使用FP32。
        AscendC::GetErfMaxMinTmpSize(
            erfShape,
            sizeof(float),
            false,
            maxTmpBytes,
            minTmpBytes);

        uint64_t tmpBytes =
            maxTmpBytes > minTmpBytes
                ? maxTmpBytes
                : minTmpBytes;

        if (tmpBytes < ALIGN_BYTES) {
            tmpBytes = ALIGN_BYTES;
        }

        tmpBytes =
            ((tmpBytes + ALIGN_BYTES - 1)
                / ALIGN_BYTES)
            * ALIGN_BYTES;

        const uint64_t queueBytes =
            2ULL
            * BUFFER_NUM
            * tileLength
            * typeSize;

        const uint64_t floatWorkBytes =
            static_cast<uint64_t>(floatBufferNum)
            * tileLength
            * sizeof(float);

        const uint64_t requiredUb =
            queueBytes + floatWorkBytes + tmpBytes;

        if (tmpBytes <= 0xFFFFFFFFULL &&
            requiredUb <= availableUb) {
            erfTmpBytes =
                static_cast<uint32_t>(tmpBytes);
            break;
        }

        // UB不足时缩小Tile，不能强行使用8192。
        if (tileLength <= alignElements) {
            return ge::GRAPH_FAILED;
        }

        tileLength =
            ((tileLength / 2) / alignElements)
            * alignElements;

        if (tileLength < alignElements) {
            tileLength = alignElements;
        }
    }

    // ------------------------------------------------------------
    // 6. 写入Tiling
    // ------------------------------------------------------------
    GeluTilingData* tiling =
        context->GetTilingData<GeluTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength =
        static_cast<uint32_t>(totalLength);

    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    // 保留字段兼容现有结构。
    // 当前Kernel不根据该字段手动申请临时空间。
    tiling->erfTmpBytes = erfTmpBytes;

    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(dtype);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    context->SetBlockDim(usedCoreNum);

    size_t* workspace =
        context->GetWorkspaceSizes(1);

    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* inputShape =
        context->GetInputShape(0);

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(
        0,
        context->GetInputDataType(0));

    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char* name)
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

} // namespace ops