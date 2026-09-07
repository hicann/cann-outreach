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

// 输入队列和输出队列均采用双缓冲：
// inputQueue 2 份 + outputQueue 2 份 = 4 份 UB buffer。
constexpr uint64_t TOTAL_UB_BUFFER_NUM = 4U;

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(
    const gert::Shape& inputShape)
{
    if (inputShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return inputShape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "AIV coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "UB size must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    // ============================================================
    // 1. 获取输入 shape
    // ============================================================
    const auto* inputStorageShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);

    const gert::Shape inputShape = EnsureNotScalar(
        inputStorageShape->GetStorageShape());

    const int64_t totalNumSigned =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNumSigned <= 0,
        OP_LOGE(
            context,
            "Square input element count must be greater than 0"),
        return ge::GRAPH_FAILED);

    const uint64_t totalNum =
        static_cast<uint64_t>(totalNumSigned);

    // ============================================================
    // 2. 获取数据类型，并设置 TilingKey
    // ============================================================
    const auto* inputDesc =
        context->GetInputDesc(0);

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
            OP_LOGE(
                context,
                "Square only supports float16 and float32");

            return ge::GRAPH_FAILED;
    }

    // ============================================================
    // 3. 获取硬件 UB 容量和 AIV 核数
    // ============================================================
    uint64_t ubSize = 0;
    int64_t platformCoreNum = 0;

    if (GetPlatformInfo(
            context,
            ubSize,
            platformCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // float32：32 / 4 = 8 个元素
    // float16：32 / 2 = 16 个元素
    const uint64_t alignNum =
        DATA_BLOCK_BYTES / typeSize;

    // ============================================================
    // 4. 根据 UB 容量计算 ubFactor
    // ============================================================
    //
    // 总共需要：
    //   inputQueue  × BUFFER_NUM(2)
    //   outputQueue × BUFFER_NUM(2)
    //
    // 所以 UB 中最多分配 4 份 tile。
    const uint64_t maxUbElements =
        ubSize / (TOTAL_UB_BUFFER_NUM * typeSize);

    // 向下按 32 字节对齐，保证 UB buffer 长度合法。
    const uint64_t ubFactor =
        (maxUbElements / alignNum) * alignNum;

    OP_CHECK_IF(
        ubFactor == 0,
        OP_LOGE(
            context,
            "UB size is too small for Square"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 5. 计算分核参数
    // ============================================================
    //
    // 一个核至少应当负责一个 32 字节数据块。
    // 如果数据量很小，则不启动无效核。
    const uint64_t maxUsefulCoreNum =
        (totalNum + alignNum - 1) / alignNum;

    const uint64_t candidateCoreNum =
        std::min<uint64_t>(
            static_cast<uint64_t>(platformCoreNum),
            maxUsefulCoreNum);

    OP_CHECK_IF(
        candidateCoreNum == 0,
        OP_LOGE(context, "Calculated coreNum is 0"),
        return ge::GRAPH_FAILED);

    // 初步计算每核平均元素数。
    const uint64_t averageBlock =
        (totalNum + candidateCoreNum - 1) /
        candidateCoreNum;

    // 每核起始位置必须保持 32 字节对齐。
    //
    // 这样只有整个张量的最终尾块可能非对齐，
    // 不同核不会同时写入同一个 32 字节数据块。
    const uint64_t blockFactor =
        ((averageBlock + alignNum - 1) / alignNum) *
        alignNum;

    // blockFactor 向上对齐后，最初计算的核数可能出现空核。
    // 因此重新计算实际需要启动的核数。
    const uint64_t usedCoreNum =
        (totalNum + blockFactor - 1) /
        blockFactor;

    OP_CHECK_IF(
        usedCoreNum == 0 ||
            usedCoreNum >
                static_cast<uint64_t>(platformCoreNum),
        OP_LOGE(
            context,
            "Invalid usedCoreNum: %lu",
            usedCoreNum),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 6. 写入 TilingData
    // ============================================================
    SquareTilingData tilingData {};

    tilingData.totalNum =
        static_cast<int64_t>(totalNum);

    tilingData.blockFactor =
        static_cast<int64_t>(blockFactor);

    tilingData.ubFactor =
        static_cast<int64_t>(ubFactor);

    auto* rawTilingData =
        context->GetRawTilingData();

    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);

    OP_CHECK_IF(
        rawTilingData->GetCapacity() <
            sizeof(SquareTilingData),
        OP_LOGE(
            context,
            "Tiling buffer capacity is insufficient"),
        return ge::GRAPH_FAILED);

    std::memcpy(
        rawTilingData->GetData(),
        &tilingData,
        sizeof(SquareTilingData));

    rawTilingData->SetDataSize(
        sizeof(SquareTilingData));

    // ============================================================
    // 7. 设置运行参数
    // ============================================================
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    context->SetTilingKey(tilingKey);

    if (GetWorkspaceSize(context) !=
        ge::GRAPH_SUCCESS) {
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
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

} // namespace optiling