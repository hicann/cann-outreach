#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BUFFER_COUNT = 4;

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

static ge::graphStatus GetShapeAndDtype(
    gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    const gert::StorageShape* inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);

    const gert::Shape inputShape = EnsureNotScalar(inputX->GetStorageShape());
    totalNum = inputShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be greater than 0"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    OP_CHECK_IF(
        dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT,
        OP_LOGE(context, "Square only supports float16 and float32"),
        return ge::GRAPH_FAILED);

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

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(
        GetShapeAndDtype(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndDtype error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    const int64_t blockFactor = CeilDiv(totalNum, usedCoreNum);
    usedCoreNum = CeilDiv(totalNum, blockFactor);

    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "invalid UB block size"), return ge::GRAPH_FAILED);

    int64_t alignElements = ubBlockSize / typeSize;
    if (alignElements < 1) {
        alignElements = 1;
    }

    const int64_t ubElements = static_cast<int64_t>(ubSize) / typeSize;
    int64_t maxUbFactor = FloorAlign(FloorDiv(ubElements, BUFFER_COUNT), alignElements);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    int64_t ubFactor = CeilAlign(blockFactor, alignElements);
    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    }
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