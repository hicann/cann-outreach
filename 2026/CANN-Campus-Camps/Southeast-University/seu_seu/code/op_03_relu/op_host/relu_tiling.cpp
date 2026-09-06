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

/*
 * 使用 4 字节作为统一的保守估计：
 *
 * float32 = 4 Byte
 * float16 = 2 Byte
 *
 * 使用 4 Byte 可以保证 FP16 / FP32 两种场景都不会超出 UB。
 */
constexpr int64_t TYPE_SIZE = 4;

/*
 * 一个 Core 至少尽量处理 1024 个元素，
 * 防止小 Tensor 盲目拉满所有 Vector Core，
 * 导致核启动开销大于计算收益。
 */
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

/*
 * Kernel:
 *
 * inputQueueX  : BUFFER_NUM = 2
 * outputQueueY : BUFFER_NUM = 2
 *
 * 总共需要：
 *
 * 2 + 2 = 4
 *
 * 块 UB Buffer。
 */
constexpr int64_t UB_BUFFER_NUM = 4;

/*
 * DataCopyPad + DataCopyParams 的 blockLen
 * 在常见 Ascend 910B / CANN 接口下使用 uint16_t，
 * 单次 blockLen 不应达到 65536 Byte。
 *
 * 这里通过 ubFactor 限制避免边界问题。
 */
constexpr int64_t MAX_COPY_BYTES = 65535;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return in_shape;
}


/*
 * 获取平台的：
 *
 * 1. UB 大小
 * 2. AIV Core 数量
 */
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
 * 本算子不需要额外 Workspace。
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
 * ReLU Tiling
 */
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // ================================================================
    // 1. 获取平台信息
    // ================================================================
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


    // ================================================================
    // 2. 获取 Workspace
    // ================================================================
    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);


    // ================================================================
    // 3. 获取输入 Shape
    // ================================================================
    auto inputX =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputX);

    /*
     * StorageShape 是 Kernel 实际处理的数据 Shape。
     *
     * Scalar 时转成 [1]。
     */
    const gert::Shape inputShape =
        EnsureNotScalar(
            inputX->GetStorageShape());

    int64_t totalNum =
        inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(
            context,
            "invalid totalNum: %ld",
            totalNum),
        return ge::GRAPH_FAILED);


    // ================================================================
    // 4. 获取 dtype
    // ================================================================
    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();

    OP_CHECK_IF(
        dataType != ge::DT_FLOAT &&
            dataType != ge::DT_FLOAT16,
        OP_LOGE(
            context,
            "Relu only supports float32/float16"),
        return ge::GRAPH_FAILED);


    // ================================================================
    // 5. 获取 TilingData
    // ================================================================
    ReluTilingData* tiling =
        context->GetTilingData<
            ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    // ================================================================
    // 6. Core 切分
    //
    // 原则：
    //
    // 每个 Core 尽量至少处理 1024 个元素。
    //
    // 对题目标准 Shape：
    //
    //     [8, 2048]
    //
    // totalNum:
    //
    //     8 * 2048 = 16384
    //
    // usedCoreNum:
    //
    //     ceil(16384 / 1024)
    //     = 16
    //
    // blockFactor:
    //
    //     16384 / 16
    //     = 1024
    //
    // ================================================================
    int64_t usedCoreNum = 1;
    int64_t blockFactor = 0;

    if (totalNum > 0) {
        int64_t coreNumByData =
            CeilDiv(
                totalNum,
                MIN_SPLIT_THRESHOLD);

        if (coreNumByData < 1) {
            coreNumByData = 1;
        }

        usedCoreNum =
            (coreNumByData < coreNum)
                ? coreNumByData
                : coreNum;

        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        blockFactor =
            CeilDiv(
                totalNum,
                usedCoreNum);

        /*
         * 根据真正的 blockFactor 再反算一次核数，
         * 确保不会启动没有数据的 Core。
         */
        usedCoreNum =
            CeilDiv(
                totalNum,
                blockFactor);
    }


    // ================================================================
    // 7. UB 切分
    //
    // Double Buffer:
    //
    // inputQueueX:
    //     buffer 0
    //     buffer 1
    //
    // outputQueueY:
    //     buffer 0
    //     buffer 1
    //
    // 总计四块 LocalTensor Buffer。
    // ================================================================
    int64_t ubCanUse =
        static_cast<int64_t>(
            ubSize);

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    OP_CHECK_IF(
        ubBlockSize <= 0,
        OP_LOGE(
            context,
            "invalid ubBlockSize"),
        return ge::GRAPH_FAILED);


    /*
     * ubSize / TYPE_SIZE
     *
     * 转为“元素数量”。
     *
     * 再除以 4，为四个 Double Buffer
     * LocalTensor 均分 UB。
     */
    int64_t ubFactor =
        FloorAlign(
            FloorDiv(
                ubCanUse / TYPE_SIZE,
                UB_BUFFER_NUM),
            ubBlockSize);


    /*
     * 限制单次 DataCopyPad 的长度。
     *
     * TYPE_SIZE=4 是保守值，所以 FP16 同样安全。
     */
    int64_t maxCopyElements =
        FloorAlign(
            MAX_COPY_BYTES /
                TYPE_SIZE,
            ubBlockSize);

    if (ubFactor >
        maxCopyElements) {
        ubFactor =
            maxCopyElements;
    }


    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(
            context,
            "invalid ubFactor: %ld",
            ubFactor),
        return ge::GRAPH_FAILED);


    // ================================================================
    // 8. 写入 TilingData
    // ================================================================
    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;


    // ================================================================
    // 9. 设置 Kernel Block 数
    // ================================================================
    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));


    // ================================================================
    // 10. 根据 dtype 设置 TilingKey
    //
    // relu.cpp:
    //
    // MODE_0 -> Relu<half>
    // MODE_1 -> Relu<float>
    // ================================================================
    uint64_t tilingKey = 0;

    if (dataType ==
        ge::DT_FLOAT16) {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(
        tilingKey);

    return ge::GRAPH_SUCCESS;
}


/*
 * 本题不需要额外 Parse 信息。
 */
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

