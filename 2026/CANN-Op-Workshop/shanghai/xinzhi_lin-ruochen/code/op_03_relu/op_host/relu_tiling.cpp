/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr uint64_t UB_BUFFER_COUNT = 4;  // Input/output, each double-buffered.
constexpr uint64_t MAX_UB_FACTOR = 4096;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu supports only float32 and float16"),
        return ge::GRAPH_FAILED);
    const auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);

    const uint64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const uint64_t alignNum = 32 / typeSize;
    const uint64_t ubCapacity = (ubSize / (UB_BUFFER_COUNT * typeSize) / alignNum) * alignNum;
    OP_CHECK_IF(ubCapacity == 0, OP_LOGE(context, "Insufficient UB memory"), return ge::GRAPH_FAILED);

    // Avoid launching many cores for small tensors. Core ranges start at 32-byte boundaries.
    const uint64_t totalLength = static_cast<uint64_t>(totalNum);
    uint64_t blockNum = 1;
    uint64_t blockFactor = 1;
    if (totalLength > 0) {
        const uint64_t usefulCores =
            (totalLength + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
        blockNum = usefulCores < static_cast<uint64_t>(coreNum)
            ? usefulCores : static_cast<uint64_t>(coreNum);
        const uint64_t alignedBlocks = (totalLength + alignNum - 1) / alignNum;
        const uint64_t blocksPerCore = (alignedBlocks + blockNum - 1) / blockNum;
        blockFactor = blocksPerCore * alignNum;
        blockFactor = blockFactor < totalLength ? blockFactor : totalLength;
        blockNum = (totalLength + blockFactor - 1) / blockFactor;
    }

    // Prefer two tiles per core for overlap, with a bounded UB allocation.
    uint64_t ubFactor = ((blockFactor + 2 * alignNum - 1) / (2 * alignNum)) * alignNum;
    ubFactor = ubFactor < MAX_UB_FACTOR ? ubFactor : MAX_UB_FACTOR;
    ubFactor = ubFactor < ubCapacity ? ubFactor : ubCapacity;
    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(blockFactor);
    tiling->ubFactor = static_cast<int64_t>(ubFactor);
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
