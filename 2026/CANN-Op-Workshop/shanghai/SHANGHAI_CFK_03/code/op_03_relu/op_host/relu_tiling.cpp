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
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Unsupported input data type %d", static_cast<int32_t>(dataType)),
        return ge::GRAPH_FAILED);

    const int64_t totalNum = EnsureNotScalar(inputStorageShape->GetShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input element count %ld", totalNum), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const uint64_t tilingKey = (dataType == ge::DT_FLOAT16)
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);

    tiling->totalNum = totalNum;
    tiling->blockFactor = 0;
    tiling->ubFactor = 0;
    if (totalNum == 0) {
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    constexpr int64_t kDataCopyAlignBytes = 32;
    constexpr int64_t kQueueCount = 2;
    constexpr int64_t kBufferNum = 2;
    constexpr int64_t kMaxAivUbSize = 192 * 1024 - 256;
    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : TYPE_SIZE;
    const int64_t elementAlign = kDataCopyAlignBytes / typeSize;
    const int64_t minElementsPerCore = (MIN_SPLIT_THRESHOLD * TYPE_SIZE + typeSize - 1) / typeSize;
    const int64_t requestedCoreNum = (totalNum + minElementsPerCore - 1) / minElementsPerCore;
    const int64_t usedCoreNum = (requestedCoreNum < coreNum) ? requestedCoreNum : coreNum;
    const int64_t averageElements = (totalNum + usedCoreNum - 1) / usedCoreNum;
    const int64_t blockFactor = ((averageElements + elementAlign - 1) / elementAlign) * elementAlign;
    const int64_t blockDim = (totalNum + blockFactor - 1) / blockFactor;

    const int64_t queueBytesPerElement = typeSize * kBufferNum * kQueueCount;
    const int64_t usableUbSize = (static_cast<int64_t>(ubSize) < kMaxAivUbSize)
        ? static_cast<int64_t>(ubSize)
        : kMaxAivUbSize;
    int64_t ubFactor = usableUbSize / queueBytesPerElement;
    ubFactor = (ubFactor / elementAlign) * elementAlign;
    OP_CHECK_IF(ubFactor < elementAlign, OP_LOGE(context, "UB is too small for Relu buffers"), return ge::GRAPH_FAILED);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
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
