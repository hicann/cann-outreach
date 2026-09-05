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

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

constexpr int64_t ALIGN_BYTES = 32;

constexpr int64_t BUFFER_NUM = 2;

constexpr int64_t QUEUE_NUM = 2;


// ================================================================
// 获取平台信息
// ================================================================

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

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(
            context,
            "coreNum is invalid"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// ================================================================
// Workspace
// ================================================================

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}


// ================================================================
// Tiling
// ================================================================

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;

    int64_t maxCoreNum = 0;

    // ------------------------------------------------------------
    // 获取平台信息
    // ------------------------------------------------------------

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            maxCoreNum) !=
            ge::GRAPH_SUCCESS,

        OP_LOGE(
            context,
            "GetPlatformInfo failed"),

        return ge::GRAPH_FAILED);


    // ------------------------------------------------------------
    // Workspace
    // ------------------------------------------------------------

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,

        OP_LOGE(
            context,
            "GetWorkspaceSize failed"),

        return ge::GRAPH_FAILED);


    // ============================================================
    // 获取 Shape
    // ============================================================

    auto inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    const gert::Shape& shape =
        inputShape->GetStorageShape();


    // ============================================================
    // 计算总元素数
    // ============================================================

    int64_t totalNum = 1;

    const size_t dimNum =
        shape.GetDimNum();

    if (dimNum == 0) {

        totalNum = 1;

    } else {

        for (size_t i = 0;
             i < dimNum;
             ++i) {

            totalNum *=
                shape.GetDim(i);
        }
    }


    OP_CHECK_IF(
        totalNum <= 0,

        OP_LOGE(
            context,
            "Invalid totalNum"),

        return ge::GRAPH_FAILED);


    // ============================================================
    // 获取 dtype
    // ============================================================

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType inputType =
        inputDesc->GetDataType();

    int64_t typeSize = 0;

    if (inputType ==
        ge::DT_FLOAT16) {

        typeSize = 2;

    } else if (inputType ==
               ge::DT_FLOAT) {

        typeSize = 4;

    } else {

        OP_LOGE(
            context,
            "Square only supports "
            "float16 and float32");

        return ge::GRAPH_FAILED;
    }


    // ============================================================
    // 32B 对齐需要多少元素
    // ============================================================

    const int64_t alignElementNum =
        ALIGN_BYTES / typeSize;


    // ============================================================
    // 动态选择 Core 数
    //
    // 小数据不要启动太多 Core
    // ============================================================

    int64_t blockDim = maxCoreNum;

    // 每个 Core 至少尽量处理一部分数据
    if (totalNum < maxCoreNum *
                    alignElementNum) {

        blockDim =
            CeilDiv(
                totalNum,
                alignElementNum);
    }

    if (blockDim <= 0) {
        blockDim = 1;
    }

    if (blockDim > maxCoreNum) {
        blockDim = maxCoreNum;
    }


    // ============================================================
    // 每个 Core 的数据量
    // ============================================================

    int64_t blockFactor =
        CeilDiv(
            totalNum,
            blockDim);

    // 32B 对齐
    blockFactor =
        CeilAlign(
            blockFactor,
            alignElementNum);

    if (blockFactor <
        alignElementNum) {

        blockFactor =
            alignElementNum;
    }


    // ============================================================
    // 重新计算实际 Core 数
    // ============================================================

    blockDim =
        CeilDiv(
            totalNum,
            blockFactor);

    if (blockDim <= 0) {
        blockDim = 1;
    }

    if (blockDim > maxCoreNum) {
        blockDim = maxCoreNum;
    }


    // ============================================================
    // 最后一个 Core 的实际数据量
    // ============================================================

    int64_t tailNum =
        totalNum -
        (blockDim - 1) *
        blockFactor;

    if (tailNum <= 0) {

        tailNum =
            blockFactor;
    }


    // ============================================================
    // UB Tile 大小
    //
    // 两个 Queue：
    //
    // input  : 2 buffers
    // output : 2 buffers
    //
    // 总共 4 份 UB
    // ============================================================

    int64_t ubFactor =
        static_cast<int64_t>(
            ubSize /
            (QUEUE_NUM *
             BUFFER_NUM *
             typeSize));


    // 32B 对齐
    ubFactor =
        FloorAlign(
            ubFactor,
            alignElementNum);


    if (ubFactor <
        alignElementNum) {

        ubFactor =
            alignElementNum;
    }


    // UB Tile 不超过一个 Core
    if (ubFactor >
        blockFactor) {

        ubFactor =
            blockFactor;
    }


    // ============================================================
    // TilingData
    // ============================================================

    SquareTilingData* tiling =
        context->GetTilingData<
            SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;

    tiling->tailNum =
        tailNum;


    // ============================================================
    // 设置 Core 数
    // ============================================================

    context->SetBlockDim(
        static_cast<uint32_t>(
            blockDim));


    // ============================================================
    // 设置 Tiling Key
    // ============================================================

    uint64_t tilingKey = 0;

    if (inputType ==
        ge::DT_FLOAT16) {

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);

    } else {

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(
        tilingKey);


    return ge::GRAPH_SUCCESS;
}


// ================================================================
// Tiling Parse
// ================================================================

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


// ================================================================
// Compile Info
// ================================================================

struct SquareCompileInfo {};


// ================================================================
// 注册
// ================================================================

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

} // namespace optiling