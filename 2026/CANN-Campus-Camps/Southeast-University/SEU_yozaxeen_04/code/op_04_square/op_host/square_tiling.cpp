/*!
 * \file square_tiling.cpp
 * \brief Square leaderboard-extreme tiling.
 *
 * Strategy:
 *   - Prefer 8 AIV cores for performance-sized elementwise workloads.
 *   - Regular cores own a 256B-aligned block, so their GM copy and vector Mul
 *     have no padding/tail path.
 *   - Only the final core may be irregular.
 *   - Keep each regular block within 255 vector repeats, allowing exactly one
 *     vector instruction in the hot path.
 */

#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t PREFERRED_CORE_NUM = 8;
constexpr int64_t MAX_VECTOR_REPEATS = 255;

constexpr uint32_t FLAG_UNIFORM_BLOCKS = 1U << 0;
constexpr uint32_t FLAG_LAST_COPY_ALIGNED = 1U << 1;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    return (inShape.GetDimNum() == 0) ? g_vec_1_shape : inShape;
}

static inline int64_t AlignUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

static ge::graphStatus GetAivCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum64 = shape.GetShapeSize();
    OP_CHECK_IF(totalNum64 <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<uint64_t>(totalNum64) > std::numeric_limits<uint32_t>::max(),
        OP_LOGE(context, "tensor is too large for this optimized tiling"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    uint64_t tilingKey = 0;
    int64_t typeSize = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
        typeSize = 4;
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    int64_t maxCoreNum = 0;
    OP_CHECK_IF(
        GetAivCoreNum(context, maxCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetAivCoreNum failed"),
        return ge::GRAPH_FAILED);

    const int64_t vectorElems = 256 / typeSize; // fp16:128, fp32:64
    const int64_t maxBlockElems = vectorElems * MAX_VECTOR_REPEATS;

    // Aggressive leaderboard choice: 8 cores is the sweet spot already
    // validated on the same 910B elementwise path by Sub/Mul.  Very small
    // tensors naturally collapse to fewer logical cores after alignment.
    int64_t requestedCoreNum = (maxCoreNum < PREFERRED_CORE_NUM)
        ? maxCoreNum : PREFERRED_CORE_NUM;
    if (requestedCoreNum < 1) requestedCoreNum = 1;

    // If 8 cores cannot keep a regular block within one vector instruction,
    // raise the core count only as much as necessary.
    const int64_t minCoreForOneShot = CeilDiv(totalNum64, maxBlockElems);
    if (requestedCoreNum < minCoreForOneShot) {
        requestedCoreNum = minCoreForOneShot;
        if (requestedCoreNum > maxCoreNum) requestedCoreNum = maxCoreNum;
    }

    int64_t blockFactor = AlignUp(CeilDiv(totalNum64, requestedCoreNum), vectorElems);

    // On extremely large inputs, increase cores until a regular block fits
    // the single-vector-instruction fast path whenever hardware allows it.
    while (blockFactor > maxBlockElems && requestedCoreNum < maxCoreNum) {
        ++requestedCoreNum;
        blockFactor = AlignUp(CeilDiv(totalNum64, requestedCoreNum), vectorElems);
    }

    OP_CHECK_IF(blockFactor <= 0, OP_LOGE(context, "blockFactor is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockFactor > maxBlockElems,
        OP_LOGE(context, "input too large for single-pass optimized kernel"),
        return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalNum64, blockFactor);
    if (usedCoreNum < 1) usedCoreNum = 1;
    OP_CHECK_IF(usedCoreNum > maxCoreNum,
        OP_LOGE(context, "usedCoreNum exceeds hardware cores"),
        return ge::GRAPH_FAILED);

    const int64_t lastBlockLength = totalNum64 - blockFactor * (usedCoreNum - 1);
    OP_CHECK_IF(lastBlockLength <= 0 || lastBlockLength > blockFactor,
        OP_LOGE(context, "lastBlockLength is invalid"),
        return ge::GRAPH_FAILED);

    const int64_t normalRepeats = blockFactor / vectorElems;
    const int64_t lastFullRepeats = lastBlockLength / vectorElems;
    const int64_t lastTail = lastBlockLength - lastFullRepeats * vectorElems;

    uint32_t flags = 0;
    if (lastBlockLength == blockFactor) flags |= FLAG_UNIFORM_BLOCKS;
    if (((lastBlockLength * typeSize) & 31LL) == 0) flags |= FLAG_LAST_COPY_ALIGNED;

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = static_cast<uint32_t>(totalNum64);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    tiling->blockNum = static_cast<uint32_t>(usedCoreNum);
    tiling->normalRepeats = static_cast<uint32_t>(normalRepeats);
    tiling->lastBlockLength = static_cast<uint32_t>(lastBlockLength);
    tiling->lastFullRepeats = static_cast<uint32_t>(lastFullRepeats);
    tiling->lastTail = static_cast<uint32_t>(lastTail);
    tiling->flags = flags;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
