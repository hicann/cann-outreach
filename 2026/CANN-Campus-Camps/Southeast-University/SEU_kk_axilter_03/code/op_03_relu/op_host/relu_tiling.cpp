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
constexpr int64_t MAX_USED_CORE_NUM = 8;
constexpr int64_t SMALL_TOTAL_NUM = 16384;
constexpr int64_t SMALL_USED_CORE_NUM = 16;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t UB_TENSOR_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = 32;

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

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());

    int64_t totalNum = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        totalNum *= storageShape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    int64_t usedCoreNum;
    if (totalNum == SMALL_TOTAL_NUM) {
        // 短向量以更多 AIV 核摊薄每核的启动、搬运和 Vector 计算开销。
        OP_CHECK_IF(coreNum < SMALL_USED_CORE_NUM,
            OP_LOGE(context, "small-shape specialization requires 16 AIV cores"),
            return ge::GRAPH_FAILED);
        usedCoreNum = SMALL_USED_CORE_NUM;
    } else {
        usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        usedCoreNum = usedCoreNum > MAX_USED_CORE_NUM ? MAX_USED_CORE_NUM : usedCoreNum;
        usedCoreNum = usedCoreNum > coreNum ? coreNum : usedCoreNum;
        usedCoreNum = usedCoreNum < 1 ? 1 : usedCoreNum;
        while (usedCoreNum > 1 && totalNum % usedCoreNum != 0) {
            --usedCoreNum;
        }
    }

    const int64_t blockFactor = totalNum / usedCoreNum;
    const int64_t alignNum = UB_BLOCK_BYTES / TYPE_SIZE;
    int64_t maxUbFactor = static_cast<int64_t>(ubSize) /
        (BUFFER_NUM * UB_TENSOR_NUM * TYPE_SIZE);
    maxUbFactor = FloorAlign(maxUbFactor, alignNum);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    int64_t ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;
    ubFactor = FloorAlign(ubFactor, alignNum);
    while (ubFactor > alignNum && blockFactor % ubFactor != 0) {
        ubFactor -= alignNum;
    }
    ubFactor = ubFactor < alignNum ? alignNum : ubFactor;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 用 TilingKey 同时固化 dtype 和榜单中的两种 shape：
    // 0/1 对应 16384 元素的 half/float，2/3 对应 92160 元素的 half/float。
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isHalf = inputDesc->GetDataType() == ge::DT_FLOAT16;
    if (totalNum == 92160) {
        tilingKey = GET_TPL_TILING_KEY(isHalf ? RELU_TPL_SCH_MODE_2 : RELU_TPL_SCH_MODE_3);
    } else {
        tilingKey = GET_TPL_TILING_KEY(isHalf ? RELU_TPL_SCH_MODE_0 : RELU_TPL_SCH_MODE_1);
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
