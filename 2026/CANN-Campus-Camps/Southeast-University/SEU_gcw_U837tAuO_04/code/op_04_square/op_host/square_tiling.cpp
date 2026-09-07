/*!
 * \file square_tiling.cpp
 * \brief Square算子Tiling实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

// 每个AI Vector Core尽量至少处理2048个元素
constexpr int64_t MIN_SPLIT_THRESHOLD = 2048;

// 优化为单Buffer
constexpr int64_t BUFFER_NUM = 1;

// 一个输入Queue和一个输出Queue
constexpr int64_t QUEUE_NUM = 2;

// Ascend C搬运对齐字节数
constexpr int64_t ALIGN_BYTES = 32;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(
            context,
            "The number of AI Vector cores is invalid"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "The UB size is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(
    gert::TilingContext* context)
{
    size_t* workspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        workspace);

    workspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    // ============================================================
    // 1. 获取平台信息
    // ============================================================
    uint64_t ubSize = 0;
    int64_t availableCoreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            availableCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 2. 设置Workspace
    // ============================================================
    OP_CHECK_IF(
        SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "SetWorkspace failed"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 3. 获取输入Shape并计算总元素数
    // ============================================================
    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    int64_t totalNum =
        inputShape
            ->GetStorageShape()
            .GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "The total number of elements must be positive"),
        return ge::GRAPH_FAILED);

    // ============================================================
    // 4. 获取数据类型
    // ============================================================
    const gert::CompileTimeTensorDesc* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType inputDtype =
        inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;

    if (inputDtype == ge::DT_FLOAT16) {
        typeSize =
            static_cast<int64_t>(sizeof(uint16_t));

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);
    } else if (inputDtype == ge::DT_FLOAT) {
        typeSize =
            static_cast<int64_t>(sizeof(float));

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(
            context,
            "Square only supports float16 and float32");

        return ge::GRAPH_FAILED;
    }

    // FP16每32字节包含16个元素
    // FP32每32字节包含8个元素
    const int64_t alignNum =
        ALIGN_BYTES / typeSize;

    // ============================================================
    // 5. 动态确定启动核心数
    //
    // 少量元素不启动过多核心，减少多核启动和同步开销。
    // ============================================================
    int64_t targetCoreNum =
        (totalNum + MIN_SPLIT_THRESHOLD - 1)
        / MIN_SPLIT_THRESHOLD;

    if (targetCoreNum < 1) {
        targetCoreNum = 1;
    }

    if (targetCoreNum > availableCoreNum) {
        targetCoreNum = availableCoreNum;
    }

    // ============================================================
    // 6. 计算每个核负责的元素数
    //
    // 除最后一个核外，每个核的长度都按32字节对齐。
    // ============================================================
    int64_t averageNum =
        (totalNum + targetCoreNum - 1)
        / targetCoreNum;

    int64_t blockFactor =
        ((averageNum + alignNum - 1)
            / alignNum)
        * alignNum;

    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 对齐后重新计算真正需要的核心数，避免启动空核
    int64_t usedCoreNum =
        (totalNum + blockFactor - 1)
        / blockFactor;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    if (usedCoreNum > availableCoreNum) {
        usedCoreNum = availableCoreNum;
    }

    // ============================================================
    // 7. 根据UB大小确定单个Tile的最大元素数
    //
    // 单Buffer时，UB中需要：
    // 1个输入Buffer + 1个输出Buffer
    // ============================================================
    int64_t maxUbFactor =
        static_cast<int64_t>(ubSize)
        / (QUEUE_NUM * BUFFER_NUM * typeSize);

    // 预留少量UB空间，避免把UB完全用满
    maxUbFactor =
        maxUbFactor * 15 / 16;

    // 向下按照32字节对齐
    maxUbFactor =
        (maxUbFactor / alignNum)
        * alignNum;

    OP_CHECK_IF(
        maxUbFactor < alignNum,
        OP_LOGE(
            context,
            "The UB size is insufficient"),
        return ge::GRAPH_FAILED);

    int64_t ubFactor =
        blockFactor < maxUbFactor
            ? blockFactor
            : maxUbFactor;

    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // ============================================================
    // 8. 写入TilingData
    // ============================================================
    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // ============================================================
    // 9. 设置Kernel启动信息
    // ============================================================
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {
};

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

} // namespace optiling