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

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BUFFER_NUM = 2;

// 一个输入 Queue + 一个输出 Queue，每个都双缓冲
constexpr int64_t QUEUE_NUM = 2;

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
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
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


static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    /*
     * =========================================================
     * 1. 获取平台信息
     * =========================================================
     */
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);


    /*
     * =========================================================
     * 2. Workspace
     * =========================================================
     */
    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);


    /*
     * =========================================================
     * 3. 获取输入 Shape
     * =========================================================
     */
    auto inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    int64_t totalNum =
        inputShape->GetOriginShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "invalid totalNum: %ld",
            totalNum),
        return ge::GRAPH_FAILED);


    /*
     * =========================================================
     * 4. 获取数据类型
     *
     * mode 0 -> half
     * mode 1 -> float
     * =========================================================
     */
    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    auto dataType =
        inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16) {

        typeSize = 2;

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_0);

    } else if (dataType == ge::DT_FLOAT) {

        typeSize = 4;

        tilingKey =
            GET_TPL_TILING_KEY(
                SQUARE_TPL_SCH_MODE_1);

    } else {

        OP_LOGE(
            context,
            "Square only supports float16 "
            "and float32");

        return ge::GRAPH_FAILED;
    }


    /*
     * =========================================================
     * 5. 计算 32 Byte 对应多少个元素
     *
     * float16:
     *     32 / 2 = 16
     *
     * float32:
     *     32 / 4 = 8
     * =========================================================
     */
    const int64_t alignNum =
        BLOCK_SIZE / typeSize;


    /*
     * =========================================================
     * 6. 确定实际使用的核数
     * =========================================================
     */
    int64_t usedCoreNum =
        (totalNum < coreNum)
            ? totalNum
            : coreNum;

    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }


    /*
     * =========================================================
     * 7. 每个核负责的数据量
     *
     * 采用连续切分：
     *
     * Core 0:
     * [0, blockFactor)
     *
     * Core 1:
     * [blockFactor, 2*blockFactor)
     *
     * ...
     * =========================================================
     */
    int64_t blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);


    /*
     * =========================================================
     * 8. 根据 UB 大小计算 tile 大小
     *
     * 输入 Queue:
     *
     *     BUFFER_NUM * ubFactor * typeSize
     *
     * 输出 Queue:
     *
     *     BUFFER_NUM * ubFactor * typeSize
     *
     * 总共：
     *
     *     2 * BUFFER_NUM
     *
     * =========================================================
     */
    const int64_t queueCount =
        BUFFER_NUM * QUEUE_NUM;

    int64_t ubFactor =
        static_cast<int64_t>(
            ubSize /
            (queueCount * typeSize));


    /*
     * =========================================================
     * 9. UB tile 按 32B 向下对齐
     * =========================================================
     */
    ubFactor =
        (ubFactor / alignNum) *
        alignNum;


    /*
     * =========================================================
     * 10. 保证至少一个 32B block
     *
     * 非对齐尾块需要 DataCopyPad，
     * 所以 UB 至少能够容纳一个 32B。
     * =========================================================
     */
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }


    /*
     * =========================================================
     * 11. 如果 blockFactor 很小，不要把 ubFactor
     *     设置成一个非对齐的 blockFactor。
     *
     * 例如 float16：
     *
     * blockFactor = 17
     *
     * 不能：
     *
     * ubFactor = 17
     *
     * 因为最后 1 个 half 进行 DataCopyPad 时，
     * 需要至少一个 32B buffer。
     *
     * 所以保持：
     *
     * ubFactor = 16
     * =========================================================
     */
    if (blockFactor >= alignNum &&
        blockFactor < ubFactor) {

        ubFactor =
            (blockFactor / alignNum) *
            alignNum;

        if (ubFactor < alignNum) {
            ubFactor = alignNum;
        }
    }


    /*
     * =========================================================
     * 12. 获取 TilingData
     * =========================================================
     */
    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    /*
     * =========================================================
     * 13. 写入 TilingData
     * =========================================================
     */
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;


    /*
     * =========================================================
     * 14. 设置 BlockDim
     * =========================================================
     */
    context->SetBlockDim(
        usedCoreNum);


    /*
     * =========================================================
     * 15. 设置 TilingKey
     *
     * float16 -> mode 0
     * float32 -> mode 1
     * =========================================================
     */
    context->SetTilingKey(
        tilingKey);


    OP_LOGI(
        context,
        "Square Tiling: "
        "totalNum=%ld, "
        "coreNum=%ld, "
        "blockFactor=%ld, "
        "ubFactor=%ld, "
        "typeSize=%ld, "
        "tilingKey=%lu",
        totalNum,
        usedCoreNum,
        blockFactor,
        ubFactor,
        typeSize,
        tilingKey);

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