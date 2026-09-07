/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

// 当前已经实测1024字节比512、768更快
constexpr int64_t MIN_SPLIT_BYTES = 1024;

// Kernel使用单Buffer
constexpr int64_t BUFFER_NUM = 1;

// 一个输入Queue和一个输出Queue
constexpr int64_t QUEUE_NUM = 2;

// GM搬运对齐字节数
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
        OP_LOGE(context, "coreNum is invalid"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
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
    uint64_t ubSize = 0;
    int64_t availableCoreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            availableCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspace failed"),
        return ge::GRAPH_FAILED);

    // 获取输入Shape
    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    // 任意多维Shape的总元素数量
    int64_t totalNum =
        inputShape->GetStorageShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum must be positive"),
        return ge::GRAPH_FAILED);

    // 获取输入类型
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
        typeSize = 2;

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);
    } else if (inputDtype == ge::DT_FLOAT) {
        typeSize = 4;

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(
            context,
            "Square only supports float16 and float32");

        return ge::GRAPH_FAILED;
    }

    // 一个32字节块包含的元素数量
    // FP16为16，FP32为8
    const int64_t alignNum =
        ALIGN_BYTES / typeSize;

    // 每核期望处理1024字节
    const int64_t minSplitThreshold =
        MIN_SPLIT_BYTES / typeSize;

    // 根据数据量决定启动核数
    int64_t targetCoreNum =
        (totalNum + minSplitThreshold - 1)
        / minSplitThreshold;

    if (targetCoreNum < 1) {
        targetCoreNum = 1;
    }

    if (targetCoreNum > availableCoreNum) {
        targetCoreNum = availableCoreNum;
    }

    // 初始每核元素数
    int64_t averageNum =
        (totalNum + targetCoreNum - 1)
        / targetCoreNum;

    // 每核长度向32字节对齐
    int64_t blockFactor =
        ((averageNum + alignNum - 1) / alignNum)
        * alignNum;

    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 对齐后重新计算实际启动核数，避免空核
    int64_t usedCoreNum =
        (totalNum + blockFactor - 1)
        / blockFactor;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    if (usedCoreNum > availableCoreNum) {
        usedCoreNum = availableCoreNum;
    }

    // 一个输入Queue和一个输出Queue，各使用一个Buffer
    int64_t maxUbFactor =
        static_cast<int64_t>(ubSize)
        / (QUEUE_NUM * BUFFER_NUM * typeSize);

    // 留出1/16 UB空间，避免队列和系统开销导致申请失败
    maxUbFactor =
        maxUbFactor * 15 / 16;

    // UB长度向下对齐到32字节
    maxUbFactor =
        (maxUbFactor / alignNum)
        * alignNum;

    OP_CHECK_IF(
        maxUbFactor < alignNum,
        OP_LOGE(context, "UB size is insufficient"),
        return ge::GRAPH_FAILED);

    int64_t ubFactor =
        blockFactor < maxUbFactor
            ? blockFactor
            : maxUbFactor;

    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    context->SetTilingKey(tilingKey);

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