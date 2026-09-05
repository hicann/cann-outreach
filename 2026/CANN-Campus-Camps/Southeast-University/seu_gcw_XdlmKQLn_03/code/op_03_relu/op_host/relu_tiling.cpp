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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t TENSOR_BUFFER_NUM = 2; // one input tensor and one output tensor

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

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid input shape size: %ld", totalNum),
                return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        typeSize = static_cast<int64_t>(sizeof(uint16_t));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else if (dataType == ge::DT_FLOAT) {
        typeSize = static_cast<int64_t>(sizeof(float));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported input dtype: %d", static_cast<int32_t>(dataType));
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // Keep each active core sufficiently busy instead of launching every AIV for a
    // small tensor. blockFactor is aligned so all non-tail cores start at an
    // address suitable for vector/DMA operations.
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        usedCoreNum = (usedCoreNum < coreNum) ? usedCoreNum : coreNum;
    }

    int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0 || ubBlockSize < typeSize,
                OP_LOGE(context, "invalid UB block size: %ld", ubBlockSize), return ge::GRAPH_FAILED);
    int64_t alignNum = ubBlockSize / typeSize;

    int64_t blockFactor = (totalNum == 0) ? alignNum : CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    if (totalNum > 0) {
        // Alignment may make the last initially selected core unnecessary.
        usedCoreNum = CeilDiv(totalNum, blockFactor);
    }

    // Double buffering needs two slots for both the input and output queues.
    int64_t totalBufferNum = DOUBLE_BUFFER_NUM * TENSOR_BUFFER_NUM;
    int64_t maxUbFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize), typeSize * totalBufferNum), alignNum);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small for Relu double buffering"),
                return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;

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
