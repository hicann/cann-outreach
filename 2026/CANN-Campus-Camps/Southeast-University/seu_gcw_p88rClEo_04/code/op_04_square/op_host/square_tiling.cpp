/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t DATA_BLOCK_BYTES = 32U;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "AIV coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "UB size must be greater than 0"),
        return ge::GRAPH_FAILED);

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
    // 1. 输入 shape 与总元素数
    const auto* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);

    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    const int64_t totalNumSigned = inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNumSigned <= 0,
        OP_LOGE(context, "Square input element count must be greater than 0"),
        return ge::GRAPH_FAILED);

    const uint64_t totalNum = static_cast<uint64_t>(totalNumSigned);

    // 2. 数据类型与 tiling key
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    uint64_t typeSize = 0;
    uint64_t tilingKey = 0;

    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            typeSize = sizeof(uint16_t);
            tilingKey = 0;
            break;
        case ge::DT_FLOAT:
            typeSize = sizeof(float);
            tilingKey = 1;
            break;
        default:
            OP_LOGE(context, "Square only supports float16 and float32");
            return ge::GRAPH_FAILED;
    }

    // 3. 硬件信息
    uint64_t ubSize = 0;
    int64_t platformCoreNum = 0;
    if (GetPlatformInfo(context, ubSize, platformCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t alignNum = DATA_BLOCK_BYTES / typeSize;

    // 4. 每核整块处理：blockFactor 向上对齐到 32 字节
    const uint64_t maxUsefulCoreNum =
        (totalNum + alignNum - 1) / alignNum;

    const uint64_t candidateCoreNum =
        std::min<uint64_t>(static_cast<uint64_t>(platformCoreNum), maxUsefulCoreNum);

    OP_CHECK_IF(
        candidateCoreNum == 0,
        OP_LOGE(context, "Calculated coreNum is 0"),
        return ge::GRAPH_FAILED);

    const uint64_t averageBlock =
        (totalNum + candidateCoreNum - 1) / candidateCoreNum;

    const uint64_t blockFactor =
        ((averageBlock + alignNum - 1) / alignNum) * alignNum;

    const uint64_t usedCoreNum =
        (totalNum + blockFactor - 1) / blockFactor;

    OP_CHECK_IF(
        usedCoreNum == 0 ||
            usedCoreNum > static_cast<uint64_t>(platformCoreNum),
        OP_LOGE(context, "Invalid usedCoreNum: %lu", usedCoreNum),
        return ge::GRAPH_FAILED);

    // 5. 写入 TilingData；ubFactor 直接给 blockFactor，kernel 里整块分配
    SquareTilingData tilingData{};
    tilingData.totalNum = static_cast<int64_t>(totalNum);
    tilingData.blockFactor = static_cast<int64_t>(blockFactor);
    tilingData.ubFactor = static_cast<int64_t>(blockFactor);

    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);

    OP_CHECK_IF(
        rawTilingData->GetCapacity() < sizeof(SquareTilingData),
        OP_LOGE(context, "Tiling buffer capacity is insufficient"),
        return ge::GRAPH_FAILED);

    std::memcpy(rawTilingData->GetData(), &tilingData, sizeof(SquareTilingData));
    rawTilingData->SetDataSize(sizeof(SquareTilingData));

    // 6. 运行参数
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(tilingKey);

    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
