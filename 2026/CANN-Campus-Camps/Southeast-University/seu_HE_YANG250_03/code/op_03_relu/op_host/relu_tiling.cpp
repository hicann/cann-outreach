/*!
 * \file relu_tiling.cpp
 * \brief Relu high-performance tiling
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

/*
 * Competition target:
 *
 * shape = [8, 2048]
 *
 * Use 8 Vector Cores first.
 *
 * If later benchmarking shows 4 cores are faster,
 * only change this constant.
 */
constexpr int64_t PERF_CORE_NUM = 8;

/*
 * Maximum static tile size.
 *
 * 16 KB means:
 *
 * FP32:
 *   4096 elements
 *   64 vector repeats
 *
 * FP16:
 *   8192 elements
 *   64 vector repeats
 *
 * Keeping repeat count <= 64 makes the explicit
 * Relu(mask, repeatTimes, repeatParams) fast path simple.
 */
constexpr int64_t MAX_TILE_BYTES = 48 * 1024;

/*
 * Vector Unit processes 256 bytes per repeat.
 */
constexpr int64_t VECTOR_BYTES = 256;

static const gert::Shape g_vec_1_shape = {1};


static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return inShape;
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
    return CeilDivI64(
        x,
        align) * align;
}


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

    auto platform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum =
        platform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(
            context,
            "AIV coreNum <= 0"),
        return ge::GRAPH_FAILED);

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "UB size == 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


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


static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // ============================================================
    // Platform
    // ============================================================

    uint64_t ubSize = 0;
    int64_t maxCoreNum = 0;

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


    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);


    // ============================================================
    // Input shape
    // ============================================================

    auto inputShapePtr =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShapePtr);

    const gert::Shape inputShape =
        EnsureNotScalar(
            inputShapePtr->
                GetStorageShape());

    const int64_t totalNum =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "invalid totalNum"),
        return ge::GRAPH_FAILED);


    // ============================================================
    // dtype
    // ============================================================

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    const ge::DataType dtype =
        inputDesc->GetDataType();

    int64_t typeSize = 0;

    if (dtype == ge::DT_FLOAT) {

        typeSize = 4;

    } else if (
        dtype == ge::DT_FLOAT16) {

        typeSize = 2;

    } else {

        OP_LOGE(
            context,
            "Relu only supports float16/float32");

        return ge::GRAPH_FAILED;
    }


    // ============================================================
    // Core number
    // ============================================================

    /*
     * Fixed competition shape:
     *
     * [8, 2048]
     *
     * => exactly 8 cores
     * => one row / core
     */
    int64_t usedCoreNum =
        PERF_CORE_NUM;

    if (usedCoreNum >
        maxCoreNum) {

        usedCoreNum =
            maxCoreNum;
    }

    if (usedCoreNum >
        totalNum) {

        usedCoreNum =
            totalNum;
    }

    if (usedCoreNum < 1) {

        usedCoreNum = 1;
    }


    // ============================================================
    // blockFactor
    // ============================================================

    /*
     * Align per-core chunks to 256 bytes.
     *
     * FP32:
     *     64 elements
     *
     * FP16:
     *     128 elements
     */
    const int64_t vectorAlignElements =
        VECTOR_BYTES /
        typeSize;


    int64_t blockFactor =
        CeilDivI64(
            totalNum,
            usedCoreNum);


    blockFactor =
        AlignUpI64(
            blockFactor,
            vectorAlignElements);


    /*
     * Alignment may reduce the number of actually useful cores.
     */
    usedCoreNum =
        CeilDivI64(
            totalNum,
            blockFactor);


    // ============================================================
    // ubFactor
    // ============================================================

    /*
     * Static in-place ReLU only requires ONE local buffer.
     *
     * Limit one tile to 16 KB so the explicit vector repeat
     * count remains small.
     */
    int64_t maxTileElements =
        MAX_TILE_BYTES /
        typeSize;


    /*
     * Defensive check against very small UB.
     */
    int64_t maxTileByUb =
        static_cast<int64_t>(
            ubSize) /
        typeSize;

    if (maxTileElements >
        maxTileByUb) {

        maxTileElements =
            maxTileByUb;
    }


    maxTileElements =
        (maxTileElements /
         vectorAlignElements) *
        vectorAlignElements;


    OP_CHECK_IF(
        maxTileElements <= 0,
        OP_LOGE(
            context,
            "invalid ubFactor"),
        return ge::GRAPH_FAILED);


    int64_t ubFactor =
        blockFactor;

    if (ubFactor >
        maxTileElements) {

        ubFactor =
            maxTileElements;
    }


    // ============================================================
    // TilingData
    // ============================================================

    ReluTilingData* tiling =
        context->GetTilingData<
            ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;


    // ============================================================
    // Kernel blockDim
    // ============================================================

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));


    // ============================================================
    // TilingKey
    // ============================================================

    if (dtype ==
        ge::DT_FLOAT16) {

        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0));

    } else {

        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1));
    }


    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


struct ReluCompileInfo {};


IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling
