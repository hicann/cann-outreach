/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t UB_BLOCK_BYTES = 32UL;
constexpr uint64_t BUFFER_NUM = 2UL;
constexpr uint64_t TENSOR_QUEUE_NUM = 2UL;

static inline uint64_t CeilDivU64(
    uint64_t value,
    uint64_t divisor)
{
    return value / divisor +
        static_cast<uint64_t>(
            value % divisor != 0);
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(
            context,
            "coreNum is not positive"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    const auto* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    uint64_t typeSize = 0;
    uint32_t schMode = 0;

    // schMode=0 编译 half 核；
    // schMode=1 编译 float 核。
    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            typeSize = sizeof(uint16_t);
            schMode = SQUARE_TPL_SCH_MODE_0;
            break;

        case ge::DT_FLOAT:
            typeSize = sizeof(float);
            schMode = SQUARE_TPL_SCH_MODE_1;
            break;

        default:
            OP_LOGE(
                context,
                "Square only supports float16 and float32");
            return ge::GRAPH_FAILED;
    }

    // Square 是逐元素算子，不需要保留各维度信息；
    // 将 ND shape 展平成连续的一维元素数量。
    const gert::Shape& storageShape =
        inputShape->GetStorageShape();

    uint64_t totalNum = 1;

    for (size_t i = 0;
         i < storageShape.GetDimNum();
         ++i) {
        const int64_t dim =
            storageShape.GetDim(i);

        OP_CHECK_IF(
            dim <= 0,
            OP_LOGE(
                context,
                "Every dimension must be positive, "
                "but dim[%zu] is %lld",
                i,
                static_cast<long long>(dim)),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            totalNum >
                static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::max()) /
                static_cast<uint64_t>(dim),
            OP_LOGE(
                context,
                "The number of input elements "
                "overflows int64_t"),
            return ge::GRAPH_FAILED);

        totalNum *=
            static_cast<uint64_t>(dim);
    }

    uint64_t ubSize = 0;
    int64_t platformCoreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            platformCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "Failed to get platform information"),
        return ge::GRAPH_FAILED);

    // 一个硬件 DataBlock 为 32B：
    // float32 对齐元素数为 8；
    // float16 对齐元素数为 16。
    const uint64_t alignNum =
        UB_BLOCK_BYTES / typeSize;

    // 小 shape 不启动无意义的空核。
    const uint64_t usefulCoreNum =
        CeilDivU64(totalNum, alignNum);

    const uint64_t blockDim =
        std::min(
            static_cast<uint64_t>(
                platformCoreNum),
            usefulCoreNum);

    OP_CHECK_IF(
        blockDim == 0,
        OP_LOGE(
            context,
            "The calculated blockDim is 0"),
        return ge::GRAPH_FAILED);

    // 每个核最多处理 blockFactor 个元素。
    // 最后一核在 kernel 中按实际 remain 截断。
    const uint64_t blockFactor =
        CeilDivU64(totalNum, blockDim);

    /*
     * inputQueue：2 个 buffer
     * outputQueue：2 个 buffer
     *
     * 因此一个 tile 最多使用总 UB 的 1/4。
     */
    const uint64_t maxUbFactor =
        (ubSize /
         (BUFFER_NUM *
          TENSOR_QUEUE_NUM *
          typeSize) /
         alignNum) *
        alignNum;

    OP_CHECK_IF(
        maxUbFactor == 0,
        OP_LOGE(
            context,
            "UB is too small for one "
            "32-byte data block"),
        return ge::GRAPH_FAILED);

    const uint64_t alignedBlockFactor =
        CeilDivU64(
            blockFactor,
            alignNum) *
        alignNum;

    const uint64_t ubFactor =
        std::min(
            alignedBlockFactor,
            maxUbFactor);

    SquareTilingData* tiling =
        context->GetTilingData<
            SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(blockDim));

    // 根据 schMode 生成与 square<0>/square<1>
    // 对应的 tilingKey。
    ASCENDC_TPL_SEL_PARAM(
        context,
        schMode);

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "Failed to set workspace size"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

} // namespace optiling