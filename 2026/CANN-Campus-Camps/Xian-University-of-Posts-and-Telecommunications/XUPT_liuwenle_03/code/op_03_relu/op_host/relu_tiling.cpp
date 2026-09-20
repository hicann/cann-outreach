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
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// ReLU：1个输入 + 1个输出
// DoubleBuffer后共需要4块UB buffer
constexpr int64_t UB_BUFFER_COUNT = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
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

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // ==============================
    // 1. 获取硬件信息
    // ==============================
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

    // ==============================
    // 2. Workspace
    // ==============================
    OP_CHECK_IF(
        GetWorkspaceSize(context)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ==============================
    // 3. 获取输入shape
    // ==============================
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

    // ==============================
    // 4. 获取dtype
    // ==============================
    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();

    int64_t typeSize = TYPE_SIZE;
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16) {

        typeSize = 2;

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);

    } else if (
        dataType == ge::DT_FLOAT) {

        typeSize = 4;

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);

    } else {

        OP_LOGE(
            context,
            "unsupported dtype");

        return ge::GRAPH_FAILED;
    }

    // ==============================
    // 5. 获取TilingData
    // ==============================
    ReluTilingData* tiling =
        context
            ->GetTilingData<
                ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    // ==============================
    // 6. 多核切分
    // ==============================

    // 避免数据量很小时开启过多核
    int64_t usedCoreNum =
        CeilDiv(
            totalNum,
            MIN_SPLIT_THRESHOLD);

    usedCoreNum =
        (usedCoreNum < 1)
            ? 1
            : usedCoreNum;

    usedCoreNum =
        (usedCoreNum > coreNum)
            ? coreNum
            : usedCoreNum;

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);

    // ==============================
    // 7. UB切分
    // ==============================

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    OP_CHECK_IF(
        ubBlockSize <= 0,
        OP_LOGE(
            context,
            "invalid ubBlockSize"),
        return ge::GRAPH_FAILED);

    // 一输入一输出 + DoubleBuffer
    // 共4块buffer
    int64_t maxUbFactor =
        FloorAlign(
            FloorDiv(
                static_cast<int64_t>(
                    ubSize) /
                    typeSize,
                UB_BUFFER_COUNT),
            ubBlockSize);

    OP_CHECK_IF(
        maxUbFactor <= 0,
        OP_LOGE(
            context,
            "UB is too small"),
        return ge::GRAPH_FAILED);

    int64_t alignedBlockFactor =
        CeilAlign(
            tiling->blockFactor,
            ubBlockSize);

    tiling->ubFactor =
        (alignedBlockFactor <
         maxUbFactor)
            ? alignedBlockFactor
            : maxUbFactor;

    // ==============================
    // 8. 设置核数和TilingKey
    // ==============================
    context->SetBlockDim(
        usedCoreNum);

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