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

#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t TENSOR_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t FAST_TOTAL_NUM = 8 * 2048;
constexpr int64_t FAST_CORE_NUM = 16;
constexpr int64_t FAST_BLOCK_FACTOR = 1024;
constexpr int64_t FAST_UB_FACTOR = 512;

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
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "input shape is invalid"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = 0;
    bool isFp16 = false;
    if (dataType == ge::DT_FLOAT16) {
        typeSize = static_cast<int64_t>(sizeof(uint16_t));
        isFp16 = true;
    } else if (dataType == ge::DT_FLOAT) {
        typeSize = static_cast<int64_t>(sizeof(float));
    } else {
        OP_LOGE(context, "unsupported input dtype: %d", static_cast<int32_t>(dataType));
        return ge::GRAPH_FAILED;
    }

    // DataCopy 的数据块和各核起始地址按 32 字节对齐。
    const int64_t alignNum = BLOCK_SIZE / typeSize;
    const int64_t maxUbFactor = FloorAlign(
        static_cast<int64_t>(ubSize / (BUFFER_NUM * TENSOR_NUM * typeSize)), alignNum);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        const int64_t targetCoreNum = std::min(
            coreNum, std::max<int64_t>(1, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD)));
        blockFactor = CeilAlign(CeilDiv(totalNum, targetCoreNum), alignNum);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        // 每核至少拆成两个 tile，使双缓冲可以重叠搬入、计算和搬出。
        const int64_t doubleBufferTile = CeilAlign(CeilDiv(blockFactor, BUFFER_NUM), alignNum);
        ubFactor = std::min(doubleBufferTile, maxUbFactor);
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    const bool useFastPath = totalNum == FAST_TOTAL_NUM && usedCoreNum == FAST_CORE_NUM &&
        blockFactor == FAST_BLOCK_FACTOR && ubFactor == FAST_UB_FACTOR;
    uint64_t tilingKey = 0;
    if (isFp16 && useFastPath) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_FP16_FAST);
    } else if (isFp16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_FP16_GENERAL);
    } else if (useFastPath) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_FP32_FAST);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_FP32_GENERAL);
    }

    context->SetBlockDim(usedCoreNum);
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
