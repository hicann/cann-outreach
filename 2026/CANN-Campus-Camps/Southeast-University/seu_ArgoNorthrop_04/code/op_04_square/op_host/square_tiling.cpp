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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
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

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 拿 totalNum：TilingContext::GetInputShape 返回 StorageShape*
    const gert::StorageShape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
    gert::Shape safe_shape = EnsureNotScalar(input_shape->GetStorageShape());
    uint64_t totalNumRaw = 1;
    for (size_t d = 0; d < safe_shape.GetDimNum(); ++d) {
        totalNumRaw *= safe_shape.GetDim(d);
    }
    tiling->totalNum = static_cast<int64_t>(totalNumRaw);

    // dtype 拿对齐系数
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = (inputDesc != nullptr) ? inputDesc->GetDataType() : ge::DT_FLOAT;
    int64_t dtypeSize = ge::GetSizeByDataType(dtype);
    int64_t alignElements = 32 / dtypeSize;  // fp32=8, fp16=16

    // 每核处理的元素数（向上对齐）
    int64_t blockFactorRaw = CeilDiv(tiling->totalNum, coreNum);
    tiling->blockFactor = CeilAlign(blockFactorRaw, alignElements);

    // 实际启动核数
    int64_t blockDim = CeilDiv(tiling->totalNum, tiling->blockFactor);
    if (blockDim < 1) blockDim = 1;
    context->SetBlockDim(blockDim);

    // UB 块大小
    constexpr int64_t UB_BLOCK_DEFAULT = 2048;
    tiling->ubFactor = CeilAlign(UB_BLOCK_DEFAULT, alignElements);

    // 根据输入 dtype 选择 tilingKey（复用上面已声明好的 inputDesc）
    uint64_t tilingKey;
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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
