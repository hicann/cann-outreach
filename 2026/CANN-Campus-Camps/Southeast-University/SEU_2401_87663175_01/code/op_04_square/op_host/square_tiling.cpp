/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现 - v4 static-tensor optimized
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t GM_PERF_ALIGN_BYTES = 512;

// v3 是 512B/core，Square 这种极轻量算子会把小 Shape 切得过碎。
// A/B tuning variant: 4KB/core。
constexpr int64_t TARGET_BYTES_PER_CORE = 4 * 1024;

// 静态 Tensor 不依赖 TPipe/TQue；预留少量 UB 余量。
constexpr int64_t UB_RESERVE_BYTES = 512;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "coreNum is invalid: %ld", coreNum),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(UB_RESERVE_BYTES),
        OP_LOGE(context, "UB is too small: %lu", ubSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static inline int64_t CeilAlignInt64(int64_t x, int64_t align)
{
    return CeilDivInt64(x, align) * align;
}

static inline int64_t FloorAlignInt64(int64_t x, int64_t align)
{
    return (x / align) * align;
}

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum = inputShape->GetOriginShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "invalid totalNum: %ld", totalNum),
        return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType dtype = inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;

    if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported dtype: %d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    const int64_t dataBlockElements = DATA_BLOCK_BYTES / typeSize;
    const int64_t perfAlignElements = GM_PERF_ALIGN_BYTES / typeSize;
    const int64_t targetElementsPerCore = TARGET_BYTES_PER_CORE / typeSize;

    // floor 可以避免仅超过阈值几个字节时额外启动一个极小尾核。
    // 正常情况下每核数据大致落在 [4KB, 8KB)；大 Shape 最终由物理核数封顶。
    int64_t desiredBlockDim = totalNum / targetElementsPerCore;
    if (desiredBlockDim < 1) {
        desiredBlockDim = 1;
    }
    if (desiredBlockDim > coreNum) {
        desiredBlockDim = coreNum;
    }

    // 核间起点强制按 512B 对齐。
    const int64_t rawBlockFactor = CeilDivInt64(totalNum, desiredBlockDim);
    const int64_t blockFactor =
        CeilAlignInt64(rawBlockFactor, perfAlignElements);

    int64_t blockDim = CeilDivInt64(totalNum, blockFactor);
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    OP_CHECK_IF(
        blockDim <= 0,
        OP_LOGE(context, "invalid blockDim: %ld", blockDim),
        return ge::GRAPH_FAILED);

    const int64_t usableUbBytes =
        static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;

    // 单核 block 能装进 UB 时，不为 DoubleBuffer 强制折半：
    // 直接一次 CopyIn + Mul + CopyOut，走最短 fast path。
    const int64_t maxSingleUbFactor =
        FloorAlignInt64(
            usableUbBytes / typeSize,
            perfAlignElements);

    OP_CHECK_IF(
        maxSingleUbFactor < dataBlockElements,
        OP_LOGE(context, "usable UB is too small: %ld", usableUbBytes),
        return ge::GRAPH_FAILED);

    int64_t ubFactor = 0;

    if (blockFactor <= maxSingleUbFactor) {
        ubFactor = blockFactor;
    } else {
        // 只有真正超大 block 才使用手工 ping-pong。
        const int64_t bytesPerBuffer = usableUbBytes / 2;
        ubFactor =
            FloorAlignInt64(
                bytesPerBuffer / typeSize,
                perfAlignElements);

        OP_CHECK_IF(
            ubFactor < dataBlockElements,
            OP_LOGE(context, "double-buffer UB is too small: %ld", bytesPerBuffer),
            return ge::GRAPH_FAILED);
    }

    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling