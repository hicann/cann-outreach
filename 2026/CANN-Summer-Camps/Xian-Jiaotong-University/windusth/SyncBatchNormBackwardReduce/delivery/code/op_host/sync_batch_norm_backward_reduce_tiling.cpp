/*!
 * \file sync_batch_norm_backward_reduce_tiling.cpp
 * \brief SyncBatchNormBackwardReduce 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_data.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_key.h"
#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BLOCK_BYTES = 32;

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

static int64_t GetShapeSize(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();
}

static int64_t GetTypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return 4;
    }
    return 2;
}

static ge::graphStatus SyncBatchNormBackwardReduceTilingFunc(gert::TilingContext* context)
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

    SyncBatchNormBackwardReduceTilingData* tiling = context->GetTilingData<SyncBatchNormBackwardReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = GetShapeSize(inputShape->GetStorageShape());
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = GetTypeSize(inputDesc->GetDataType());
    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_0);
            break;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_1);
            break;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_2);
            break;
        default:
            OP_LOGE(context, "Unsupported dtype for SyncBatchNormBackwardReduce");
            return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    tiling->totalNum = totalNum;
    int64_t minBlockFactor = BLOCK_BYTES / typeSize;
    int64_t rawBlockFactor = totalNum > 0 ? CeilDiv(totalNum, coreNum) : 1;
    // Keep multi-core GM write ranges on 32B boundaries. Splitting tiny channel vectors into
    // one-element chunks can make adjacent cores write different elements in the same cache line.
    tiling->blockFactor = CeilAlign(std::max(rawBlockFactor, minBlockFactor), minBlockFactor);
    int64_t bytesPerElement = typeSize * BUFFER_NUM * 6;
    if (inputDesc->GetDataType() != ge::DT_FLOAT) {
        bytesPerElement += static_cast<int64_t>(sizeof(float)) * 6;
    }
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElement, minBlockFactor);
    tiling->ubFactor = ubFactor > 0 ? ubFactor : minBlockFactor;

    context->SetBlockDim(totalNum > 0 ? CeilDiv(totalNum, tiling->blockFactor) : 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSyncBatchNormBackwardReduce([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SyncBatchNormBackwardReduceCompileInfo {};

IMPL_OP_OPTILING(SyncBatchNormBackwardReduce).Tiling(SyncBatchNormBackwardReduceTilingFunc).TilingParse<SyncBatchNormBackwardReduceCompileInfo>(TilingParseForSyncBatchNormBackwardReduce);

} // namespace optiling
