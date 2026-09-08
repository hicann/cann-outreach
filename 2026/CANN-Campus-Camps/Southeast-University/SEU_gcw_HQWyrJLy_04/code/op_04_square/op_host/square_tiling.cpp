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

// 不需要额外 Workspace
constexpr uint32_t WS_SYS_SIZE = 0U;

// Double Buffer
constexpr int64_t BUFFER_NUM = 2;

// Square 需要一个输入 + 一个输出
constexpr int64_t TENSOR_NUM = 2;

// 小数据时没必要启动很多核
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;


// ============================================================
// 获取平台信息
// ============================================================

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

    // 获取 Vector Core 数量
    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "coreNum is invalid"),
        return ge::GRAPH_FAILED);

    // 获取 UB 大小
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// ============================================================
// Workspace
// ============================================================

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


// ============================================================
// Square Tiling
// ============================================================

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    // --------------------------------------------------------
    // 1. 获取硬件信息
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // 2. Workspace
    // --------------------------------------------------------

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,

        OP_LOGE(
            context,
            "GetWorkspaceSize error"),

        return ge::GRAPH_FAILED);


    // --------------------------------------------------------
    // 3. 获取输入 Tensor
    // --------------------------------------------------------

    const gert::Tensor* inputTensor =
        context->GetRequiredInputTensor(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputTensor);


    // --------------------------------------------------------
    // 4. 得到整个 Tensor 的元素总数
    //
    // 不假设固定 shape。
    //
    // 例如：
    // (8, 2048) -> 16384
    // (3, 17)   -> 51
    // (2,5,37)  -> 370
    // --------------------------------------------------------

    int64_t totalNum =
        static_cast<int64_t>(
            inputTensor->GetShapeSize());

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "totalNum is invalid"),

        return ge::GRAPH_FAILED);


    // --------------------------------------------------------
    // 5. 获取数据类型
    // --------------------------------------------------------

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();


    int64_t typeSize = 0;

    if (dataType == ge::DT_FLOAT16) {

        // half = 2 Bytes
        typeSize = 2;

    } else if (dataType == ge::DT_FLOAT) {

        // float = 4 Bytes
        typeSize = 4;

    } else {

        OP_LOGE(
            context,
            "Square only supports float16 and float32");

        return ge::GRAPH_FAILED;
    }


    // --------------------------------------------------------
    // 6. 获取 TilingData
    // --------------------------------------------------------

    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    // --------------------------------------------------------
    // 7. 决定启动多少个 AI Vector Core
    //
    // 每大约 1024 个元素增加一个核，
    // 但不能超过硬件提供的核数。
    // --------------------------------------------------------

    int64_t usedCoreNum =
        (totalNum +
         MIN_SPLIT_THRESHOLD - 1) /
        MIN_SPLIT_THRESHOLD;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // usedCoreNum 不会超过元素总数
    if (usedCoreNum > totalNum) {
        usedCoreNum = totalNum;
    }


    // --------------------------------------------------------
    // 8. 每个 Core 最多处理多少元素
    //
    // ceil(totalNum / usedCoreNum)
    // --------------------------------------------------------

    int64_t blockFactor =
        (totalNum +
         usedCoreNum - 1) /
        usedCoreNum;


    // --------------------------------------------------------
    // 9. UB Tiling
    //
    // Square Kernel 中存在：
    //
    // inputQueueX
    // outputQueueY
    //
    // 两个 Tensor。
    //
    // 每个 Queue 又使用 Double Buffer：
    //
    // 2 Tensor × 2 Buffer
    //
    // 所以 UB 大致分成 4 份。
    // --------------------------------------------------------

    int64_t maxUbElements =
        static_cast<int64_t>(ubSize) /
        typeSize /
        TENSOR_NUM /
        BUFFER_NUM;


    // --------------------------------------------------------
    // 10. 计算 32B 对齐对应的元素数量
    //
    // float32:
    // 32 / 4 = 8 elements
    //
    // float16:
    // 32 / 2 = 16 elements
    // --------------------------------------------------------

    int64_t alignNum =
        32 / typeSize;


    // --------------------------------------------------------
    // UB Tile 本身保持 32B 对齐
    // --------------------------------------------------------

    int64_t ubFactor =
        (maxUbElements / alignNum) *
        alignNum;


    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(
            context,
            "UB is too small"),

        return ge::GRAPH_FAILED);


    // --------------------------------------------------------
    // 如果一个 Core 的数据量本身就很小，
    // UB 没必要开得特别大。
    //
    // 这里是“向上对齐”，非常重要。
    //
    // 例如 float32:
    // blockFactor = 13
    //
    // 对齐后：
    // ubFactor = 16
    //
    // UB 内存仍然满足 32B 对齐，
    // 但真实计算只处理 13 个。
    // --------------------------------------------------------

    if (ubFactor > blockFactor) {

        ubFactor =
            ((blockFactor +
              alignNum - 1) /
             alignNum) *
            alignNum;
    }


    // --------------------------------------------------------
    // 11. 写入 TilingData
    // --------------------------------------------------------

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;


    // --------------------------------------------------------
    // 12. 设置 Kernel 启动核数
    // --------------------------------------------------------

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));


    // --------------------------------------------------------
    // 13. 根据 dtype 选择 Kernel
    //
    // MODE 0 -> half
    // MODE 1 -> float
    // --------------------------------------------------------

    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16) {

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


// ============================================================
// Tiling Parse
// ============================================================

static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


struct SquareCompileInfo {};


// 注册 Tiling
IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);


} // namespace optiling