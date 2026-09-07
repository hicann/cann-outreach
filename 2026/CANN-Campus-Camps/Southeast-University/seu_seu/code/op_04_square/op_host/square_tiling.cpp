/*!
 * \file square_tiling.cpp
 * \brief Square optimized tiling
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

/*
 * Performance tuning parameter.
 *
 * 16KB/core means:
 *
 * FP32:
 *     4096 elements/core
 *     64 vector repeats
 *
 * FP16:
 *     8192 elements/core
 *     64 vector repeats
 */
constexpr int64_t TARGET_BYTES_PER_CORE = 16 * 1024;

/*
 * Vector unit consumes 256 bytes per repeat.
 */
constexpr int64_t VECTOR_BYTES = 256;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return shape;
}

static inline int64_t CeilDivI64(
    int64_t x,
    int64_t y)
{
    return (x + y - 1) / y;
}

static inline int64_t AlignUpI64(
    int64_t x,
    int64_t align)
{
    return CeilDivI64(x, align) * align;
}


static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfo);

    auto platform =
        platform_ascendc::PlatformAscendC(
            platformInfo);

    coreNum =
        platform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(
            context,
            "invalid AIV core num"),
        return ge::GRAPH_FAILED);

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "invalid UB size"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* workspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        workspace);

    workspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t maxCoreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            maxCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);


    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * Shape
     * ============================================================
     */
    auto inputX =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputX);

    const gert::Shape inputShape =
        EnsureNotScalar(
            inputX->GetStorageShape());

    int64_t totalNum =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(
            context,
            "invalid totalNum"),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * dtype
     * ============================================================
     */
    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dtype =
        inputDesc->GetDataType();

    int64_t typeSize;

    if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        OP_LOGE(
            context,
            "Square only supports fp16/fp32");

        return ge::GRAPH_FAILED;
    }


    /*
     * ============================================================
     * UB tile
     * ============================================================
     *
     * One tile contains TARGET_BYTES_PER_CORE input data.
     *
     * input + output:
     *
     *     16KB + 16KB
     *
     * so UB consumption is very small on 910B.
     */
    int64_t tileBytes =
        TARGET_BYTES_PER_CORE;

    /*
     * Defensive fallback for unusually small UB.
     */
    int64_t maxTileByUb =
        static_cast<int64_t>(
            ubSize / 4);

    if (tileBytes > maxTileByUb) {
        tileBytes =
            maxTileByUb;
    }


    tileBytes =
        (tileBytes /
         VECTOR_BYTES) *
        VECTOR_BYTES;

    OP_CHECK_IF(
        tileBytes <= 0,
        OP_LOGE(
            context,
            "tileBytes <= 0"),
        return ge::GRAPH_FAILED);


    /*
     * Number of elements held in one static UB tile.
     */
    int64_t ubFactor =
        tileBytes /
        typeSize;


    /*
     * 256B alignment in element units.
     *
     * FP32 = 64
     * FP16 = 128
     */
    int64_t vectorAlignElements =
        VECTOR_BYTES /
        typeSize;


    /*
     * ============================================================
     * Core count
     * ============================================================
     *
     * Desired:
     *
     * one core for every ~16KB input data.
     *
     * Examples:
     *
     * 16384 FP32:
     *
     *     input = 64KB
     *     => 4 cores
     *
     * 16384 FP16:
     *
     *     input = 32KB
     *     => 2 cores
     */
    int64_t usedCoreNum = 1;

    if (totalNum > 0) {

        usedCoreNum =
            CeilDivI64(
                totalNum,
                ubFactor);

        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        if (usedCoreNum >
            maxCoreNum) {
            usedCoreNum =
                maxCoreNum;
        }
    }


    /*
     * ============================================================
     * Block factor
     * ============================================================
     */
    int64_t blockFactor = 1;

    if (totalNum > 0) {

        blockFactor =
            CeilDivI64(
                totalNum,
                usedCoreNum);


        /*
         * Align every normal core to one complete Vector repeat.
         *
         * Only the last core may contain a tail.
         */
        blockFactor =
            AlignUpI64(
                blockFactor,
                vectorAlignElements);


        /*
         * Avoid launching empty cores after alignment.
         */
        usedCoreNum =
            CeilDivI64(
                totalNum,
                blockFactor);

        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
    }


    /*
     * ============================================================
     * TilingData
     * ============================================================
     */
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


    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));


    /*
     * ============================================================
     * dtype compile-time dispatch
     * ============================================================
     */
    if (dtype ==
        ge::DT_FLOAT16) {

        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0));

    } else {

        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1));
    }


    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


struct SquareCompileInfo {};


IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);

} // namespace optiling
