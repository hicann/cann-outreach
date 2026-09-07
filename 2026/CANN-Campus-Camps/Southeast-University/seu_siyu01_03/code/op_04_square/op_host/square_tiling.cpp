/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

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
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    const gert::Shape inputShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    uint64_t totalNum = 1;
    for (size_t index = 0; index < inputShape.GetDimNum(); ++index) {
        OP_CHECK_IF(inputShape.GetDim(index) <= 0, OP_LOGE(context, "input dimension must be positive"),
                    return ge::GRAPH_FAILED);
        totalNum *= static_cast<uint64_t>(inputShape.GetDim(index));
    }
    OP_CHECK_IF(inputShape.GetDim(inputShape.GetDimNum() - 1) > 10240,
                OP_LOGE(context, "the last dimension must not exceed 10240"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(0));
    const gert::Shape outputShape = EnsureNotScalar(context->GetOutputShape(0)->GetStorageShape());
    OP_CHECK_IF(outputShape != inputShape, OP_LOGE(context, "input and output shapes must match"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to get platform information"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(inputDesc->GetDataType() != outputDesc->GetDataType(),
                OP_LOGE(context, "input and output data types must match"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputDesc->GetDataType() != ge::DT_FLOAT && inputDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(context, "only float and float16 are supported"), return ge::GRAPH_FAILED);
    const bool isFloat = inputDesc->GetDataType() == ge::DT_FLOAT;
    const uint32_t elementSize = isFloat ? sizeof(float) : sizeof(uint16_t);
    constexpr uint32_t bufferNum = 2;
    constexpr uint32_t tensorNum = 2;
    const uint64_t maxUbElements = ubSize / (bufferNum * tensorNum * elementSize);
    const uint64_t alignElements = 32 / elementSize;
    OP_CHECK_IF(maxUbElements < alignElements, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    const uint64_t ubFactor = FloorDiv(maxUbElements, alignElements) * alignElements;
    const uint64_t minElementsPerCore = isFloat ? 1024 : 512;
    const uint64_t usefulBlockNum = CeilDiv(totalNum, minElementsPerCore);
    const uint64_t blockNum = std::min<uint64_t>(static_cast<uint64_t>(coreNum), usefulBlockNum);
    const uint64_t blockFactor = CeilDiv(totalNum, blockNum);

    SquareTilingData tilingData;
    tilingData.totalNum = totalNum;
    tilingData.blockFactor = blockFactor;
    tilingData.ubFactor = ubFactor;
    OP_CHECK_IF(context->GetRawTilingData() == nullptr, OP_LOGE(context, "tiling data is null"),
                return ge::GRAPH_FAILED);
    auto* rawTilingData = context->GetRawTilingData();
    rawTilingData->SetDataSize(sizeof(tilingData));
    std::memcpy(rawTilingData->GetData(), &tilingData, sizeof(tilingData));
    context->SetBlockDim(blockNum);
    context->SetTilingKey(isFloat ? SQUARE_TPL_SCH_MODE_1 : SQUARE_TPL_SCH_MODE_0);
    return GetWorkspaceSize(context);
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc).TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
