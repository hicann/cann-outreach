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

    // 1. 输入总元素数(任意多维张量展平处理, 标量按1维处理)
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape xShape = EnsureNotScalar(inputShape->GetOriginShape());
    int64_t totalNum = xShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 2. 按输入 dtype 确定字长和 32B 对齐元素数(fp32 为 8, fp16 为 16)
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = ge::GetSizeByDataType(inputDesc->GetDataType());
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "typeSize is invalid"), return ge::GRAPH_FAILED);
    const int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;

    // 3. 多核切分: 数据量过小则单核处理, 否则均分到所有 AIV 核
    int64_t usedCoreNum = (totalNum < MIN_SPLIT_THRESHOLD) ? 1 : coreNum;
    // 每核处理量向上取整并按 32B 对齐, 保证各核 GM 偏移和 DataCopy 对齐
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    // 按对齐后的 blockFactor 重算实际核数, 去掉分不到数据的空核
    usedCoreNum = CeilDiv(totalNum, blockFactor);

    // 4. UB 内切分: 输入+输出两个队列, 各 BUFFER_NUM(2) 块 buffer
    int64_t ubFactorMax = FloorAlign(static_cast<int64_t>(ubSize) / (2 * 2 * typeSize), alignNum);
    OP_CHECK_IF(ubFactorMax <= 0, OP_LOGE(context, "ub size is too small"), return ge::GRAPH_FAILED);
    // 单核数据一个 tile 能放下就不拆分, 否则按 UB 容量上限拆分
    int64_t ubFactor = (blockFactor < ubFactorMax) ? blockFactor : ubFactorMax;

    // 5. 写回 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
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
