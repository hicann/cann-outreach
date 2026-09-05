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
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t TENSOR_NUM = 2; 

constexpr int64_t TARGET_TILE_BYTES = 16 * 1024;

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu only supports float32 and float16"), return ge::GRAPH_FAILED);

    const uint32_t DT_X = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = storageShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "input shape contains an invalid dimension"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    const ge::graphStatus platformStatus = GetPlatformInfo(context, ubSize, coreNum);
    if (platformStatus != ge::GRAPH_SUCCESS) {
        return platformStatus;
    }

    const int64_t typeSize = dataType == ge::DT_FLOAT ? TYPE_SIZE : sizeof(uint16_t);
    const int64_t ubBlockBytes = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(ubBlockBytes <= 0 || ubBlockBytes % typeSize != 0,
        OP_LOGE(context, "invalid UB block size"), return ge::GRAPH_FAILED);
    const int64_t alignNum = ubBlockBytes / typeSize;

    // 每核至少分配约 MIN_SPLIT_THRESHOLD 个元素，避免小张量过度切核。
    const int64_t wantedCoreNum = totalNum == 0
        ? 1
        : CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    const int64_t maxUsedCoreNum = wantedCoreNum < coreNum ? wantedCoreNum : coreNum;
    const int64_t blockFactor = totalNum == 0
        ? 0
        : CeilAlign(CeilDiv(totalNum, maxUsedCoreNum), alignNum);
    const int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    // 双缓冲时 UB 同时容纳 input/output 各 BUFFER_NUM 块。
    const int64_t bytesPerElementInUb = typeSize * TENSOR_NUM * BUFFER_NUM;
    const int64_t maxUbFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize), bytesPerElementInUb), alignNum);
    OP_CHECK_IF(maxUbFactor < alignNum,
        OP_LOGE(context, "UB is too small for Relu double buffering"), return ge::GRAPH_FAILED);
    const int64_t targetUbFactor = FloorAlign(FloorDiv(TARGET_TILE_BYTES, typeSize), alignNum);
    const int64_t tileLimit = targetUbFactor < maxUbFactor ? targetUbFactor : maxUbFactor;
    OP_CHECK_IF(tileLimit < alignNum,
        OP_LOGE(context, "tile size is smaller than a UB block"), return ge::GRAPH_FAILED);
    // Balance the tiles to avoid a large full tile followed by a very small tail.
    // tileNum counts actual CopyIn/Compute/CopyOut iterations, not buffer pairs.
    const int64_t tileNum = blockFactor == 0 ? 1 : CeilDiv(blockFactor, tileLimit);
    const int64_t ubFactor = blockFactor == 0
        ? alignNum
        : CeilAlign(CeilDiv(blockFactor, tileNum), alignNum);

    auto* tilingData = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->totalNum = static_cast<uint64_t>(totalNum);
    tilingData->blockFactor = static_cast<uint64_t>(blockFactor);
    tilingData->ubFactor = static_cast<uint64_t>(ubFactor);

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return GetWorkspaceSize(context);
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
