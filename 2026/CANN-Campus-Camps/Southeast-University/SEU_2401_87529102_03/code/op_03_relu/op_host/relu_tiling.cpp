/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <algorithm>
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FLOAT16_SIZE = 2;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t IO_QUEUE_NUM = 2;
constexpr int64_t MAX_TILE_BYTES = 64 * 1024;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
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
    uint64_t ubSize;
    int64_t coreNum;
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

    const gert::Tensor* input = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    const auto dtype = input->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Relu only supports float16 and float32"),
                return ge::GRAPH_FAILED);
    const int64_t totalNum = input->GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid input shape"), return ge::GRAPH_FAILED);

    // DataCopy operates on 32-byte data blocks. Keep every per-core range and
    // every UB tile aligned so that only the final transfer can be unaligned.
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? FLOAT16_SIZE : FLOAT32_SIZE;
    const int64_t ubBlockSize = GetUbBlockSize(context);
    const int64_t alignNum = ubBlockSize / typeSize;

    // UB contains one input queue and one output queue. Each queue owns two
    // buffers, which enables ping-pong execution of MTE2/Vector/MTE3.
    const int64_t usableUb = static_cast<int64_t>(ubSize) / (IO_QUEUE_NUM * DOUBLE_BUFFER_NUM);
    const int64_t tileBytes = FloorAlign(std::min(usableUb, MAX_TILE_BYTES), ubBlockSize);
    OP_CHECK_IF(tileBytes < ubBlockSize, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    const int64_t ubFactor = tileBytes / typeSize;

    // Do not launch more cores than there are aligned data blocks. For small
    // tensors, also avoid assigning less than MIN_ELEMENTS_PER_CORE on
    // average, because launch/synchronization overhead dominates ReLU itself.
    const int64_t totalBlocks = CeilDiv(totalNum, alignNum);
    const int64_t maxCoresByWork = std::max<int64_t>(1, totalNum / MIN_ELEMENTS_PER_CORE);
    const int64_t targetBlockDim = std::max<int64_t>(1, std::min({coreNum, totalBlocks, maxCoresByWork}));
    const int64_t blocksPerCore = totalBlocks == 0 ? 0 : CeilDiv(totalBlocks, targetBlockDim);
    // Rounding blocksPerCore upward can make the last launched core empty.
    // Reduce blockDim to the actual number of non-empty ranges.
    const int64_t blockDim = totalBlocks == 0 ? 1 : CeilDiv(totalBlocks, blocksPerCore);
    const int64_t blockFactor = blocksPerCore * alignNum;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
