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

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t FP16_TYPE_SIZE = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 1 输入 + 1 输出，DoubleBuffer 各 2 块，共 4 块 UB
constexpr int64_t UB_TENSOR_NUM = 4;
constexpr int64_t DEFAULT_UB_BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER_TILE_NUM = 2;

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    const gert::Shape xShape = EnsureNotScalar(inputX->GetStorageShape());
    const int64_t totalNum = xShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = TYPE_SIZE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        typeSize = FP16_TYPE_SIZE;
    }

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t ubBlockSize = GetUbBlockSize(context);
    if (ubBlockSize <= 0) {
        ubBlockSize = DEFAULT_UB_BLOCK_SIZE;
    }
    int64_t alignNum = ubBlockSize / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // 优先多核切分；过小数据避免切核开销
    int64_t usedCoreNum = coreNum;
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        usedCoreNum = 1;
    }
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // UB 切分：按 DoubleBuffer 预留 4 块 tensor，并尽量切成 2 个 tile 形成流水
    const int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t maxUbCount = FloorAlign(FloorDiv(ubCanUse / typeSize, UB_TENSOR_NUM), alignNum);
    if (maxUbCount < alignNum) {
        maxUbCount = alignNum;
    }

    int64_t ubFactor = maxUbCount;
    if (blockFactor <= maxUbCount) {
        if (blockFactor >= alignNum * DOUBLE_BUFFER_TILE_NUM) {
            ubFactor = CeilAlign(CeilDiv(blockFactor, DOUBLE_BUFFER_TILE_NUM), alignNum);
            if (ubFactor > maxUbCount) {
                ubFactor = maxUbCount;
            }
        } else {
            ubFactor = CeilAlign(blockFactor, alignNum);
            if (ubFactor > maxUbCount) {
                ubFactor = maxUbCount;
            }
        }
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
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
