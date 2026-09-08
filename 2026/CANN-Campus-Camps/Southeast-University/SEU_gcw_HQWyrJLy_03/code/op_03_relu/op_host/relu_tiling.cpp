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


// ============================================================
// 常量定义
// ============================================================

// 本题不需要额外 Workspace
constexpr uint32_t WS_SYS_SIZE = 0U;

// Double Buffer
constexpr int64_t BUFFER_NUM = 2;

// ReLU 一共需要两个 Tensor：
// input x + output y
constexpr int64_t TENSOR_NUM = 2;


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
// ReLU Tiling
// ============================================================

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // --------------------------------------------------------
    // 1. 获取 NPU 硬件信息
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
    // 2. 配置 Workspace
    // --------------------------------------------------------

    OP_CHECK_IF(
        GetWorkspaceSize(
            context) != ge::GRAPH_SUCCESS,

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
    // 4. 获取元素总数
    //
    // 本题：
    //
    // shape = (8, 2048)
    //
    // totalNum =
    // 8 * 2048
    // = 16384
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
    // 5. 获取输入数据类型
    // --------------------------------------------------------

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();


    // float16 = 2 Byte
    // float32 = 4 Byte

    int64_t typeSize = 4;

    if (dataType == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (dataType == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        OP_LOGE(
            context,
            "unsupported dtype");

        return ge::GRAPH_FAILED;
    }


    // --------------------------------------------------------
    // 6. 获取 TilingData
    // --------------------------------------------------------

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    // --------------------------------------------------------
    // 7. 选择实际使用的核数
    //
    // 本题 Tensor 第一维为 8，
    // 优先使用 8 个核：
    //
    // 16384 / 8 = 2048
    //
    // 每个核刚好处理 2048 个元素。
    //
    // 同时兼容核数不足的情况。
    // --------------------------------------------------------

    int64_t usedCoreNum = 1;

    if (coreNum >= 8) {
        usedCoreNum = 8;
    } else if (coreNum >= 4) {
        usedCoreNum = 4;
    } else if (coreNum >= 2) {
        usedCoreNum = 2;
    }


    // --------------------------------------------------------
    // 8. Core Tiling
    // --------------------------------------------------------

    int64_t blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);


    // --------------------------------------------------------
    // 9. UB Tiling
    //
    // ReLU 需要：
    //
    // inputQueueX
    // outputQueueY
    //
    // 又因为使用 Double Buffer：
    //
    // 2 Tensor * 2 Buffer = 4 块 UB
    // --------------------------------------------------------

    int64_t ubCanUse =
        static_cast<int64_t>(ubSize);

    int64_t maxUbElements =
        ubCanUse /
        typeSize /
        TENSOR_NUM /
        BUFFER_NUM;


    // --------------------------------------------------------
    // Ascend 数据搬运以 32B 为基本对齐单位。
    //
    // float32：
    // 32 / 4 = 8 个元素
    //
    // float16：
    // 32 / 2 = 16 个元素
    // --------------------------------------------------------

    int64_t alignNum =
        32 / typeSize;


    // 向下对齐
    int64_t ubFactor =
        FloorAlign(
            maxUbElements,
            alignNum);


    // 一个 Tile 没必要比一个 Core 的任务还大
    if (ubFactor > blockFactor) {
        ubFactor =
            FloorAlign(
                blockFactor,
                alignNum);
    }


    // 防止异常情况下得到 0
    if (ubFactor <= 0) {
        ubFactor =
            alignNum;
    }


    // --------------------------------------------------------
    // 10. 写入 TilingData
    // --------------------------------------------------------

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;


    // --------------------------------------------------------
    // 11. 设置启动核数
    // --------------------------------------------------------

    context->SetBlockDim(
        usedCoreNum);


    // --------------------------------------------------------
    // 12. 根据 dtype 设置 TilingKey
    //
    // MODE 0 -> half
    // MODE 1 -> float
    // --------------------------------------------------------

    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16) {

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


// ============================================================
// Tiling Parse
// ============================================================

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


struct ReluCompileInfo {};


// 注册 Tiling
IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);


} // namespace optiling