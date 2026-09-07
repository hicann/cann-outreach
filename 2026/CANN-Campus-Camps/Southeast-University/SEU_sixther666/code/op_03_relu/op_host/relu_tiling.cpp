/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

// 按 float32 计算 UB，占用更大。
// float16 使用该值时只是更加保守，不会造成 UB 越界。
constexpr int64_t TYPE_SIZE = 4;

// 希望每个核至少处理约 1024 个元素。
// 本题 totalNum = 8 * 2048 = 16384，
// 因而通常使用 16 个核。
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// Kernel 中有：
// inputQueueX  : DoubleBuffer = 2
// outputQueueY : DoubleBuffer = 2
//
// 所以总计需要考虑 4 份 UB Tensor。
constexpr int64_t UB_BUFFER_NUM = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}


/*
 * 获取平台信息：
 * 1. 可用 AIV Core 数量
 * 2. UB 大小
 */
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

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


/*
 * ReLU 不需要额外 Workspace。
 */
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


/*
 * Relu Tiling
 *
 * 主要完成：
 *
 * 1. 获取 totalNum
 * 2. 核间切分，得到 blockFactor
 * 3. UB 内切分，得到 ubFactor
 * 4. 设置 BlockDim
 * 5. 根据 dtype 设置 TilingKey
 */
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    /*
     * ============================================================
     * 1. 获取硬件信息
     * ============================================================
     */
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * 2. Workspace
     * ============================================================
     */
    OP_CHECK_IF(
        GetWorkspaceSize(context)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * 3. 获取输入 Shape
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

    const int64_t totalNum =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "invalid totalNum: %ld",
            totalNum),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * 4. 获取 TilingData
     * ============================================================
     */
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    /*
     * ============================================================
     * 5. 核间 Tiling
     * ============================================================
     *
     * 本题：
     *
     * shape = (8, 2048)
     *
     * totalNum = 16384
     *
     * 以每个核大约处理 >= 1024 元素为目标：
     *
     * usedCoreNum = ceil(16384 / 1024)
     *             = 16
     *
     * blockFactor = 16384 / 16
     *             = 1024
     *
     * 如果硬件可用核数不足，则使用实际可用核数。
     */
    int64_t usedCoreNum =
        CeilDiv(
            totalNum,
            MIN_SPLIT_THRESHOLD);

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }


    /*
     * 对当前比赛 Shape，优先选择能够整除 totalNum
     * 的核数量。
     *
     * 这样每个 Core 处理完全相同的数据量，
     * 避免最后一个 Core 出现非对齐尾块。
     */
    while (usedCoreNum > 1 &&
           totalNum % usedCoreNum != 0) {
        --usedCoreNum;
    }


    tiling->totalNum = totalNum;

    tiling->blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);


    /*
     * ============================================================
     * 6. UB Tiling
     * ============================================================
     *
     * Kernel:
     *
     * inputQueueX  -> BUFFER_NUM = 2
     * outputQueueY -> BUFFER_NUM = 2
     *
     * 总共考虑 4 个 UB buffer。
     *
     * TYPE_SIZE 使用 float32 的 4 Byte，
     * 因此同时兼容 float32 / float16。
     */
    const int64_t ubBlockSize =
        GetUbBlockSize(context);

    OP_CHECK_IF(
        ubBlockSize <= 0,
        OP_LOGE(
            context,
            "invalid ubBlockSize"),
        return ge::GRAPH_FAILED);


    const int64_t ubCanUse =
        static_cast<int64_t>(ubSize);


    int64_t maxUbFactor =
        FloorAlign(
            FloorDiv(
                ubCanUse / TYPE_SIZE,
                UB_BUFFER_NUM),
            ubBlockSize);


    OP_CHECK_IF(
        maxUbFactor <= 0,
        OP_LOGE(
            context,
            "invalid maxUbFactor"),
        return ge::GRAPH_FAILED);


    /*
     * tile 没必要超过单个 Core 的数据长度。
     */
    if (maxUbFactor >
        tiling->blockFactor) {

        tiling->ubFactor =
            tiling->blockFactor;
    } else {

        tiling->ubFactor =
            maxUbFactor;
    }


    /*
     * ============================================================
     * 7. DoubleBuffer 优化
     * ============================================================
     *
     * 如果整个 block 可以一次放进 UB，
     * 那么只有一个 tile，
     * DoubleBuffer 就发挥不了太大作用。
     *
     * 因此把单核数据拆成大约两块。
     *
     * 对本题典型结果：
     *
     * blockFactor = 1024
     * ubFactor    = 512
     *
     * 每个 Core:
     *
     * Tile0 = 512
     * Tile1 = 512
     */
    if (tiling->ubFactor ==
            tiling->blockFactor &&
        tiling->blockFactor >=
            2 * ubBlockSize) {

        int64_t halfBlock =
            FloorAlign(
                tiling->blockFactor / 2,
                ubBlockSize);

        if (halfBlock > 0) {
            tiling->ubFactor =
                halfBlock;
        }
    }


    /*
     * 安全检查
     */
    OP_CHECK_IF(
        tiling->ubFactor <= 0,
        OP_LOGE(
            context,
            "invalid ubFactor"),
        return ge::GRAPH_FAILED);


    /*
     * ============================================================
     * 8. 设置实际使用的 Core 数量
     * ============================================================
     */
    context->SetBlockDim(
        usedCoreNum);


    /*
     * ============================================================
     * 9. 根据 dtype 设置 TilingKey
     * ============================================================
     *
     * 保留原模板已有的 schMode 分配：
     *
     * float16 / bfloat16 -> MODE 0
     * float32            -> MODE 1
     */
    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    uint64_t tilingKey = 0;

    if (inputDesc->GetDataType() ==
            ge::DT_FLOAT16 ||
        inputDesc->GetDataType() ==
            ge::DT_BF16) {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);

    } else if (
        inputDesc->GetDataType() ==
            ge::DT_FLOAT) {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);

    } else {

        OP_LOGE(
            context,
            "unsupported dtype");

        return ge::GRAPH_FAILED;
    }


    context->SetTilingKey(
        tilingKey);


    return ge::GRAPH_SUCCESS;
}


/*
 * Tiling Parse
 *
 * 本算子不需要额外 CompileInfo。
 */
static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


struct ReluCompileInfo {};


/*
 * 注册 Tiling
 */
IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);


} // namespace optiling