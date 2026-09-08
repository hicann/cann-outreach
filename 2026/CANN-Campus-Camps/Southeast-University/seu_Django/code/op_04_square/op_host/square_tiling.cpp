/*!
 * \file square_tiling.cpp
 * \brief Square: aligned multicore tiling and double-buffer UB allocation.
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr uint64_t UB_RESERVE_BYTES = 8192;
constexpr uint64_t MAX_TILE_BYTES = 16384;
constexpr uint64_t BUFFER_COUNT = 4; // input x 2 + output y 2

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                      uint64_t& ubSize, int64_t& coreNum)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum <= 0 || ubSize <= UB_RESERVE_BYTES,
                OP_LOGE(context, "Invalid AIV count or UB size"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS ||
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto* desc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Square only supports FP16 and FP32"),
                return ge::GRAPH_FAILED);
    const auto& shape = inputShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE(context, "Tiling requires concrete dimensions"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dim != 0 && totalNum >
                    (std::numeric_limits<int64_t>::max() - 32) / dim,
                    OP_LOGE(context, "Tensor element count overflow"),
                    return ge::GRAPH_FAILED);
        totalNum *= dim;
    }
    // Rank-zero tensor is one scalar; a zero dimension means empty tensor.
    OP_CHECK_IF(totalNum < 0 || totalNum > std::numeric_limits<int64_t>::max() - 32,
                OP_LOGE(context, "Invalid tensor size"), return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignNum = 32 / typeSize;
    const int64_t desiredCores = std::min(coreNum,
        std::max<int64_t>(1, totalNum / MIN_SPLIT_THRESHOLD));
    const int64_t perCore = totalNum / desiredCores +
        (totalNum % desiredCores != 0);
    const int64_t blockFactor = std::max(alignNum,
        ((perCore + alignNum - 1) / alignNum) * alignNum);
    const int64_t usedCores = totalNum == 0 ? 1 :
        totalNum / blockFactor + (totalNum % blockFactor != 0);

    // Four equally sized buffers plus reserved UB must fit on the device.
    const uint64_t tileBytes = std::min(MAX_TILE_BYTES,
        (ubSize - UB_RESERVE_BYTES) / BUFFER_COUNT) / 32 * 32;
    OP_CHECK_IF(tileBytes == 0, OP_LOGE(context, "Insufficient UB"),
                return ge::GRAPH_FAILED);
    const int64_t ubFactor = std::min(blockFactor,
        static_cast<int64_t>(tileBytes / typeSize));

    auto* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    auto* rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(SquareTilingData));
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ?
        GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0) :
        GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
struct SquareCompileInfo {};
IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(TilingParseForSquare);
} // namespace optiling
