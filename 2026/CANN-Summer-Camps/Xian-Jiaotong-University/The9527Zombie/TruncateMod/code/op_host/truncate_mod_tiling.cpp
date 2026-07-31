/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod host tiling.
 *
 * Steps:
 *   1. Read platform info (AIV core count, UB size) and the input dtype.
 *   2. Build the broadcast output shape and, for each input, the element stride
 *      to walk it in the output coordinate space (0 == broadcast on that dim).
 *   3. Split the output element range across cores (32B-block granularity, the
 *      non-aligned remainder is absorbed by the last core) and choose a UB tile
 *      size that fits the per-element buffer budget.
 */
#include "register/op_def_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"

namespace optiling {

constexpr uint32_t BYTES_PER_BLOCK = 32u;
constexpr uint64_t UB_RESERVED_BYTES = 8192u;  // headroom for tiling struct / stack
constexpr uint32_t INPUT_X1_IDX = 0u;
constexpr uint32_t INPUT_X2_IDX = 1u;
// Per output element UB budget (bytes):
//   queues x1 + x2 + y : 3 * dtypeSize * bufferNum
//   fp32 compute buffers: enough for the int32 high-precision path (worst case)
constexpr uint64_t FP32_CALC_BUFS = 6u;                        // x1f, x2f, quot, tmp, split-hi, split-lo
constexpr uint64_t CALC_BYTES = FP32_CALC_BUFS * sizeof(float);

// Map ge::DataType to (tiling key, element byte size). Returns false if unsupported.
static bool SelectSchMode(ge::DataType dtype, uint64_t& tilingKey, uint32_t& dtypeSize)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            dtypeSize = 2u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_FP16);
            return true;
        case ge::DT_FLOAT:
            dtypeSize = 4u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_FP32);
            return true;
        case ge::DT_BF16:
            dtypeSize = 2u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_BF16);
            return true;
        case ge::DT_INT32:
            dtypeSize = 4u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_INT32);
            return true;
        case ge::DT_INT8:
            dtypeSize = 1u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_INT8);
            return true;
        case ge::DT_UINT8:
            dtypeSize = 1u;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATE_MOD_SCH_UINT8);
            return true;
        default:
            return false;
    }
}

// Fill outShape[] (broadcast of x1/x2) and the per-input broadcast strides.
static void BuildBroadcastInfo(const gert::Shape* x1Shape, const gert::Shape* x2Shape, TruncateModTilingData* tiling,
                               uint64_t& totalCount, uint64_t& x1Count, uint64_t& x2Count)
{
    const uint32_t rank1 = static_cast<uint32_t>(x1Shape->GetDimNum());
    const uint32_t rank2 = static_cast<uint32_t>(x2Shape->GetDimNum());
    uint32_t outRank = (rank1 > rank2) ? rank1 : rank2;
    if (outRank == 0u) {
        outRank = 1u; // treat a pure scalar as a 1-element vector
    }
    const uint32_t pad1 = outRank - ((rank1 > outRank) ? outRank : rank1);
    const uint32_t pad2 = outRank - ((rank2 > outRank) ? outRank : rank2);

    uint64_t s1[TRUNCATE_MOD_MAX_DIM];
    uint64_t s2[TRUNCATE_MOD_MAX_DIM];
    totalCount = 1u;
    x1Count = 1u;
    x2Count = 1u;
    for (uint32_t i = 0u; i < outRank; ++i) {
        s1[i] = (i < pad1) ? 1u : static_cast<uint64_t>(x1Shape->GetDim(i - pad1));
        s2[i] = (i < pad2) ? 1u : static_cast<uint64_t>(x2Shape->GetDim(i - pad2));
        const uint64_t outDim = (s1[i] > s2[i]) ? s1[i] : s2[i];
        tiling->outShape[i] = outDim;
        totalCount *= (outDim == 0u) ? 0u : outDim;
        x1Count *= s1[i];
        x2Count *= s2[i];
    }
    // Contiguous strides of each input in its own (right-aligned) layout; a
    // broadcast dim (size 1 against a larger output dim) gets stride 0.
    uint64_t c1 = 1u;
    uint64_t c2 = 1u;
    for (int32_t i = static_cast<int32_t>(outRank) - 1; i >= 0; --i) {
        tiling->x1Stride[i] = (s1[i] == 1u) ? 0u : c1;
        tiling->x2Stride[i] = (s2[i] == 1u) ? 0u : c2;
        c1 *= s1[i];
        c2 *= s2[i];
    }
    tiling->dimNum = outRank;
}

static ge::graphStatus TruncateModTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "TruncateMod tiling starts.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum == 0u, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0u;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0u, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    auto x1Desc = context->GetInputDesc(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    ge::DataType dtype = x1Desc->GetDataType();
    OP_CHECK_IF(dtype != x2Desc->GetDataType(), OP_LOGE(context, "x1 and x2 dtype must be consistent."),
                return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0u;
    uint32_t dtypeSize = 0u;
    OP_CHECK_IF(!SelectSchMode(dtype, tilingKey, dtypeSize),
                OP_LOGE(context, "unsupported dtype for TruncateMod (phase 1: bf16/fp16/fp32/int32/int8/uint8)."),
                return ge::GRAPH_FAILED);

    auto x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x2Shape = context->GetInputShape(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = TruncateModTilingData{};

    uint64_t totalCount = 0u;
    uint64_t x1Count = 0u;
    uint64_t x2Count = 0u;
    BuildBroadcastInfo(&x1Shape->GetStorageShape(), &x2Shape->GetStorageShape(), tiling, totalCount, x1Count, x2Count);
    OP_CHECK_IF(totalCount == 0u, OP_LOGE(context, "output element count must not be 0."), return ge::GRAPH_FAILED);

    tiling->x1SameShape = (x1Count == totalCount) ? 1u : 0u;
    tiling->x2SameShape = (x2Count == totalCount) ? 1u : 0u;
    tiling->x1Scalar = (x1Count == 1u) ? 1u : 0u;
    tiling->x2Scalar = (x2Count == 1u) ? 1u : 0u;

    const uint64_t blockElem = BYTES_PER_BLOCK / dtypeSize;
    tiling->blockElem = blockElem;
    tiling->totalCount = totalCount;

    // UB tile size: floor to a whole number of 32B blocks.
    uint32_t bufferNum = 2u;
    uint64_t usable = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    uint64_t perElem = 3u * static_cast<uint64_t>(dtypeSize) * bufferNum + CALC_BYTES;
    uint64_t tileCount = (usable / perElem / blockElem) * blockElem;
    if (tileCount == 0u) {
        tileCount = blockElem;
    }

    // Core split over output blocks; remainder elements go to the last core.
    uint64_t totalBlocks = totalCount / blockElem;
    uint64_t tailElems = totalCount % blockElem;
    uint64_t coreNum = aivCoreNum;
    if (totalBlocks < coreNum) {
        coreNum = (totalBlocks == 0u) ? 1u : totalBlocks;
    }
    uint64_t perCoreBlocks = totalBlocks / coreNum;
    uint64_t tailBlocks = totalBlocks % coreNum; // leading cores taking one extra block
    uint64_t perCoreCount = perCoreBlocks * blockElem;

    tiling->coreNum = coreNum;
    tiling->bufferNum = bufferNum;
    tiling->tileCount = tileCount;
    tiling->perCoreCount = perCoreCount;
    tiling->tailCoreNum = tailBlocks;
    // Last core (index coreNum-1) is never in the leading tail group, so it owns
    // the base block count plus the non-aligned remainder.
    tiling->lastCoreCount = perCoreCount + tailElems;

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0u;
    context->SetTilingKey(tilingKey);

    OP_LOGD(context,
            "TruncateMod tiling: key=%lu core=%lu total=%lu tile=%lu perCore=%lu tailCore=%lu last=%lu dim=%u "
            "x1Same=%u x2Same=%u x1Sca=%u x2Sca=%u",
            tilingKey, coreNum, totalCount, tileCount, perCoreCount, tailBlocks, tiling->lastCoreCount, tiling->dimNum,
            tiling->x1SameShape, tiling->x2SameShape, tiling->x1Scalar, tiling->x2Scalar);
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateMod)
    .Tiling(TruncateModTilingFunc)
    .TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
