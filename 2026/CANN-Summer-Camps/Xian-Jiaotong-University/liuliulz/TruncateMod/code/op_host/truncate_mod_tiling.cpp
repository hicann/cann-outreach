/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 2048;

static inline int64_t Align64(int64_t x) { return (x + 63) / 64 * 64; }

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

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const gert::Shape& s = outShape->GetStorageShape();
    int64_t totalNum = s.GetShapeSize();
    if (totalNum < 1) totalNum = 1;

    int64_t blockDim = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
    if (blockDim > coreNum) blockDim = coreNum;
    if (blockDim < 1) blockDim = 1;

    int64_t blockFactor = Align64((totalNum + blockDim - 1) / blockDim);
    if (blockFactor < 64) blockFactor = 64;

    int64_t dtypeSize = 2;
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT) dtypeSize = 4;

    int64_t ubFactor = (static_cast<int64_t>(ubSize) - 2048) / (6 * dtypeSize + 82);
    ubFactor = Align64(ubFactor);
    if (ubFactor < 64) ubFactor = 64;
    if (ubFactor < 64) ubFactor = 64;
    if (ubFactor > blockFactor) ubFactor = blockFactor;

    tiling->totalNum    = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor    = ubFactor;

    // 广播参数
    auto s1 = context->GetInputShape(0);
    auto s2 = context->GetInputShape(1);
    if (s1 != nullptr) {
        const gert::Shape& sh1 = s1->GetStorageShape();
        tiling->x1Total = sh1.GetShapeSize();
        tiling->x1LastDim = (sh1.GetDimNum() > 0) ? sh1.GetDim(sh1.GetDimNum() - 1) : 1;
    } else {
        tiling->x1Total = totalNum;
        tiling->x1LastDim = 1;
    }
    if (s2 != nullptr) {
        const gert::Shape& sh2 = s2->GetStorageShape();
        tiling->x2Total = sh2.GetShapeSize();
        tiling->x2LastDim = (sh2.GetDimNum() > 0) ? sh2.GetDim(sh2.GetDimNum() - 1) : 1;
    } else {
        tiling->x2Total = totalNum;
        tiling->x2LastDim = 1;
    }
    tiling->outLastDim = (s.GetDimNum() > 0) ? s.GetDim(s.GetDimNum() - 1) : 1;

    context->SetBlockDim(blockDim);

    uint64_t tilingKey;
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_FP16);
    } else if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_BF16);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_FP32);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

IMPL_OP_OPTILING(TruncateMod).Tiling(TilingFunc).TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
