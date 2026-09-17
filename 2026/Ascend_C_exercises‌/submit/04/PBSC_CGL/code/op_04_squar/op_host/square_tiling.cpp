/*!
 * \file square_tiling.cpp
 * \brief Square tiling implementation
 */

#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

/*
 * 竞速版使用 Single Buffer：
 *
 * inputQueueX : 1
 * outputQueueY: 1
 *
 * 所以 UB 一共只需要两份。
 */
constexpr int64_t UB_BUFFER_NUM = 2;

/*
 * 保持上一版已经验证 PASS 的安全上限。
 *
 * 65504 = 65536 - 32
 */
constexpr int64_t MAX_COPY_BYTES = 65504;

static const gert::Shape g_vec_1_shape = {1};


static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return inShape;
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

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(
            context,
            "coreNum is 0"),
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


static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    // ========================================================
    // 1. Platform
    // ========================================================

    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);


    // ========================================================
    // 2. Shape
    // ========================================================

    auto inputX =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputX);

    auto inputShape =
        EnsureNotScalar(
            inputX->GetStorageShape());

    int64_t totalNum =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "invalid input shape"),
        return ge::GRAPH_FAILED);


    // ========================================================
    // 3. dtype / TilingKey
    // ========================================================

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;

    /*
     * square.cpp：
     *
     * mode 0 = half
     * mode 1 = float
     */
    if (dataType == ge::DT_FLOAT16) {

        typeSize =
            sizeof(uint16_t);

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);

    } else if (
        dataType == ge::DT_FLOAT) {

        typeSize =
            sizeof(float);

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);

    } else {

        OP_LOGE(
            context,
            "Square supports only "
            "float16 and float32");

        return ge::GRAPH_FAILED;
    }

    context->SetTilingKey(
        tilingKey);


    // ========================================================
    // 4. 32 Byte 对齐
    // ========================================================

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    int64_t alignNum =
        ubBlockSize /
        typeSize;


    // ========================================================
    // 5. Core 数竞速优化
    //
    // 很小的数据不启动大量 Core。
    //
    // 目标：
    // 减少微秒级算子的 Core startup 开销。
    // ========================================================

    int64_t targetCoreNum =
        coreNum;

    if (totalNum <= 2048) {

        targetCoreNum = 1;

    } else if (
        totalNum <= 8192) {

        targetCoreNum =
            (coreNum < 4)
                ? coreNum
                : 4;

    } else if (
        totalNum <= 32768) {

        targetCoreNum =
            (coreNum < 8)
                ? coreNum
                : 8;

    } else if (
        totalNum <= 131072) {

        targetCoreNum =
            (coreNum < 16)
                ? coreNum
                : 16;
    }


    /*
     * 每 Core 数据量。
     *
     * 按 32B 对齐，保证除了最后一个 Core
     * 之外，绝大部分 GM 地址和长度都处于
     * DataCopy 快路径。
     */
    int64_t blockFactor =
        CeilDiv(
            totalNum,
            targetCoreNum);

    blockFactor =
        CeilAlign(
            blockFactor,
            alignNum);


    /*
     * 由于 blockFactor 被向上对齐，
     * 实际需要的 Core 数可能略小于
     * targetCoreNum。
     */
    int64_t usedCoreNum =
        CeilDiv(
            totalNum,
            blockFactor);

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));


    // ========================================================
    // 6. UB Factor
    //
    // Single Buffer：
    //
    // input  : 1份
    // output : 1份
    //
    // 所以除2。
    // ========================================================

    int64_t perBufferBytes =
        FloorDiv(
            static_cast<int64_t>(
                ubSize),
            UB_BUFFER_NUM);

    perBufferBytes =
        FloorAlign(
            perBufferBytes,
            ubBlockSize);


    /*
     * 保持 DataCopyPad 参数安全范围。
     */
    if (perBufferBytes >
        MAX_COPY_BYTES) {

        perBufferBytes =
            MAX_COPY_BYTES;
    }


    int64_t ubFactor =
        perBufferBytes /
        typeSize;


    /*
     * 当前 Core 数据一次能装入 UB 时，
     * 强制让整个 Core 只跑一次。
     *
     * 这是竞速的主要 Fast Path。
     */
    if (ubFactor >
        blockFactor) {

        ubFactor =
            blockFactor;
    }


    if (ubFactor <= 0) {

        ubFactor =
            alignNum;
    }


    /*
     * 保证 UB Factor 本身保持 32B 对齐。
     */
    ubFactor =
        FloorAlign(
            ubFactor,
            alignNum);

    if (ubFactor <= 0) {

        ubFactor =
            alignNum;
    }


    // ========================================================
    // 7. Tiling Data
    // ========================================================

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


    // ========================================================
    // 8. Workspace
    // ========================================================

    OP_CHECK_IF(
        GetWorkspaceSize(
            context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

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