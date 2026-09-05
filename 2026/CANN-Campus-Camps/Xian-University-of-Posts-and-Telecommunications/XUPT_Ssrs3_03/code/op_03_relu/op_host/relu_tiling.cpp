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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 与 op_kernel/relu.h 中的 BUFFER_NUM 保持一致
constexpr int64_t RELU_BUFFER_NUM = 2;
// UB 空间利用率（预留 10% 余量防止栈/对齐溢出）
constexpr int64_t UB_UTILIZATION_NUM = 9;
constexpr int64_t UB_UTILIZATION_DEN = 10;

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
    // ========== 1. 获取平台信息 ==========
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

    // ========== 2. 计算总元素数 ==========
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    tiling->totalNum = totalNum;

    // ========== 3. 根据 dtype 决定元素字节数 ==========
    auto inputDesc = context->GetInputDesc(0);
    int64_t typeSize = TYPE_SIZE;
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = 2;
    }

    // ========== 4. 计算 blockFactor（按 32B 对齐） ==========
    const int64_t alignNum = 32 / typeSize;   // float32: 8 / float16: 16
    int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
    blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
    int64_t usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;
    tiling->blockFactor = blockFactor;

    // ========== 5. 计算 ubFactor（优化点：UB 利用率从 80% 提升到 90%，减少循环次数） ==========
    const int64_t usableUb = static_cast<int64_t>(ubSize) * UB_UTILIZATION_NUM / UB_UTILIZATION_DEN;
    const int64_t maxUbFactor = usableUb / (2 * RELU_BUFFER_NUM * typeSize);
    int64_t ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
    ubFactor = (ubFactor / alignNum) * alignNum;
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }
    tiling->ubFactor = ubFactor;

    // ========== 6. 设置核数 ==========
    context->SetBlockDim(usedCoreNum);

    // ========== 7. tilingKey ==========
    uint64_t tilingKey;
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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
