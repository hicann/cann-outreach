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
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t ALIGN_NUM = 16;
constexpr int64_t BUFFER_NUM_HOST = 2;
constexpr int64_t TENSOR_NUM = 2;

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
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "dtype is invalid"), return ge::GRAPH_FAILED);

    int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
    blockFactor = ((blockFactor + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    int64_t usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;

    int64_t ubFactor = static_cast<int64_t>(ubSize) /
        (BUFFER_NUM_HOST * TENSOR_NUM * typeSize);
    ubFactor = (ubFactor / ALIGN_NUM) * ALIGN_NUM;
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
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
