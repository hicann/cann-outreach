/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_example_tiling.cpp
 * \brief ReluExample Tiling 实现（arch22：ascend910b/ascend910_93）
 *
 * 通用一元 element-wise Tiling：
 * - 支持任意 shape（包括 4D [N4,N3,N2,N1] 等）
 * - 按可用核数多核均分
 * - UB 按最大容量切分循环
 * - 数据量大时自动启用双缓冲
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/relu_example_tiling_data.h"
#include "../../op_kernel/arch22/relu_example_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 2;  // sizeof(half) = 2
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 2;   // 单缓冲: 1输入 + 1输出
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 4;   // 双缓冲: 2×1输入 + 2×1输出
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;     // 双缓冲启用阈值

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx, ge::DataType* dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    *totalIdx = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
        OP_LOGE(context, "relu: unsupported dtype, only DT_FLOAT16 supported"),
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

static ge::graphStatus ReluExampleTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Shape & 数据类型
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. TilingData
    ReluExampleTilingData* tiling = context->GetTilingData<ReluExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReluExampleTilingData), 0, sizeof(ReluExampleTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0);
        return ge::GRAPH_SUCCESS;
    }

    // 多核均分，按 DMA 最小粒度 (32B) 对齐
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB 切分
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    tiling->ubFactor = FloorAlign(
        FloorDiv((static_cast<int64_t>(ubSize) / TYPE_SIZE), bufferNum), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5. TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReluExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluExampleCompileInfo {};

IMPL_OP_OPTILING(ReluExample).Tiling(ReluExampleTilingFunc)
    .TilingParse<ReluExampleCompileInfo>(TilingParseForReluExample);

} // namespace optiling
