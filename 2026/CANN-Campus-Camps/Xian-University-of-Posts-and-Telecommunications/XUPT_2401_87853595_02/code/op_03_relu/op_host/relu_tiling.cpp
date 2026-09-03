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

// 按 float32 做最保守的 UB 空间计算
constexpr int64_t TYPE_SIZE = 4;

// input 双缓冲2份 + output 双缓冲2份
constexpr int64_t UB_BUFFER_NUM = 4;

// 每个Core尽量至少处理1024个元素
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return in_shape;
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

    currentWorkspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // ========================================================
    // 获取平台信息
    // ========================================================

    uint64_t ubSize;
    int64_t coreNum;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // ========================================================
    // Workspace
    // ========================================================

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ========================================================
    // 获取输入shape
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
            "totalNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    // ========================================================
    // 获取dtype
    // ========================================================

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
            "Relu only supports float32 and float16"),
        return ge::GRAPH_FAILED);

    // ========================================================
    // 获取tiling结构
    // ========================================================

    ReluTilingData* tiling =
        context->GetTilingData<
            ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    // ========================================================
    // Core切分
    // ========================================================

    int64_t maxUsedCore =
        CeilDiv(
            totalNum,
            MIN_SPLIT_THRESHOLD);

    int64_t usedCoreNum =
        coreNum < maxUsedCore
            ? coreNum
            : maxUsedCore;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    int64_t blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);

    // ========================================================
    // UB切分
    // ========================================================

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    int64_t ubCanUse =
        static_cast<int64_t>(
            ubSize);

    int64_t ubFactor =
        FloorAlign(
            FloorDiv(
                ubCanUse / TYPE_SIZE,
                UB_BUFFER_NUM),
            ubBlockSize);

    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(
            context,
            "ubFactor is invalid"),
        return ge::GRAPH_FAILED);

    // 不需要让UB块大于一个Core的数据
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // ========================================================
    // 设置tiling data
    // ========================================================

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));

    // ========================================================
    // 设置TilingKey
    // ========================================================

    uint64_t tilingKey;

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
