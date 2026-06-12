/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ... (License header)
 */

/*!
 * \file relu_tiling.cpp
 * \brief ReLU Tiling 实现（arch22）
 *
 * 一元 Elementwise 算子，纯 FP16 计算。
 * 相比 Abs 增加 1 个 zeros TBuf，因此 bufferDivisor 略大。
 *
 * bufferDivisor = (queueCount + 1) * sizeof(half)
 *   queueCount: 单缓冲=2, 双缓冲=4
 *   +1: zeros TBuf
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/relu_tiling_data.h"
#include "../../op_kernel/arch22/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FP16_BYTES = 2;
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t SINGLE_BUF_QUEUE_COUNT = 2;   // 1 in + 1 out
constexpr int64_t DOUBLE_BUF_QUEUE_COUNT = 4;   // 2 in + 2 out
constexpr int64_t ZEROS_BUF_COUNT = 1;           // zeros TBuf (always 1, no double)
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    return (inShape.GetDimNum() == 0) ? gert::Shape{gert::ShapeSize(1)} : inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeInfo(gert::TilingContext* context, int64_t* totalIdx, ge::DataType* dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "Relu: input/output shape mismatch: x=%ld, y=%ld",
                inputShapeX.GetShapeSize(), outShapeY.GetShapeSize()),
        return ge::GRAPH_FAILED);

    *totalIdx = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
                OP_LOGE(context, "Relu: unsupported dtype, only float16 supported"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 1. Platform
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Shape / dtype
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeDtypeInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeDtypeInfo error"), return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. TilingData
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReluTilingData), 0, sizeof(ReluTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;

    // 多核切分
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB 切分 — ReLU 需额外 1 个 zeros TBuf
    // bufferDivisor = (queueCount + zerosBufCount) * sizeof(half)
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t queueCount = useDoubleBuffer ? DOUBLE_BUF_QUEUE_COUNT : SINGLE_BUF_QUEUE_COUNT;
    int64_t bufferDivisor = (queueCount + ZEROS_BUF_COUNT) * FP16_BYTES;
    tiling->ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / bufferDivisor, ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5. TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
