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

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FLOAT_SIZE = 4;
constexpr int64_t HALF_SIZE = 2;
constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t QUEUE_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr uint64_t UB_RESERVED_BYTES = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // Square is elementwise, so an arbitrary ND tensor can be flattened.
    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = storageShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    const ge::DataType inputDtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (inputDtype == ge::DT_FLOAT16) {
        typeSize = HALF_SIZE;
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
    } else if (inputDtype == ge::DT_FLOAT) {
        typeSize = FLOAT_SIZE;
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Square only supports float16 and float32");
        return ge::GRAPH_FAILED;
    }

    const int64_t alignNum = DATA_BLOCK_BYTES / typeSize;

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = alignNum;
        tiling->ubFactor = alignNum;
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // Small tensors avoid excessive launch overhead; large tensors scale up
    // to the available number of Vector Cores.
    int64_t useCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (useCoreNum < 1) {
        useCoreNum = 1;
    }
    if (useCoreNum > coreNum) {
        useCoreNum = coreNum;
    }

    // Regular core segments are aligned to 32B. The last segment is allowed
    // to be shorter and is handled by DataCopyPad on the kernel side.
    int64_t blockFactor = CeilDiv(totalNum, useCoreNum);
    blockFactor = CeilAlign(blockFactor, alignNum);

    // 2 queues * 2 buffers implement a double-buffer pipeline.
    uint64_t usableUbSize = ubSize;
    if (usableUbSize > UB_RESERVED_BYTES) {
        usableUbSize -= UB_RESERVED_BYTES;
    }

    int64_t maxUbFactor =
        static_cast<int64_t>(usableUbSize) / (QUEUE_NUM * BUFFER_NUM * typeSize);
    maxUbFactor = FloorAlign(maxUbFactor, alignNum);
    OP_CHECK_IF(
        maxUbFactor < alignNum,
        OP_LOGE(context, "UB is too small for Square"),
        return ge::GRAPH_FAILED);

    // Prefer about two tiles when one core's data fits in UB, so DoubleBuffer
    // is effective. Large segments automatically use more iterations.
    int64_t targetUbFactor = CeilDiv(blockFactor, BUFFER_NUM);
    targetUbFactor = CeilAlign(targetUbFactor, alignNum);

    int64_t ubFactor = targetUbFactor;
    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(useCoreNum));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc).TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling



