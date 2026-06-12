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
 * \file atanh_tiling.cpp
 * \brief Atanh 算子 Tiling 实现
 *
 * 多核切分，UB 空间分配
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/atanh_tiling_data.h"
#include "../op_kernel/atanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 2;               // float16 = 2 bytes
constexpr size_t WORKSPACE_NUM = 1;
// atanh 需要缓冲区: 1输入 + 1输出 + 3临时(ones, tmp1, tmp2) = 5
// 单缓冲: 5, 双缓冲: 2*(1in+1out) + 3tmp = 7
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 5;
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 7;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "Atanh: input and output shape size mismatch: x=%ld, y=%ld",
            inputShapeX.GetShapeSize(), outShapeY.GetShapeSize()),
        return ge::GRAPH_FAILED);

    *totalIdx = inputShapeX.GetShapeSize();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
        OP_LOGE(context, "Atanh: invalid dtype, only float16 supported"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AtanhTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    AtanhTilingData* tiling = context->GetTilingData<AtanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AtanhTilingData), 0, sizeof(AtanhTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);

    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    // 临时缓冲区从 VECCALC 分配，不占用队列缓冲区名额
    // 单缓冲: 1in + 1out = 2, 双缓冲: 2*1in + 2*1out = 4
    int64_t queueTensorCount = useDoubleBuffer ? 4 : 2;
    tiling->ubFactor = FloorAlign(FloorDiv((static_cast<int64_t>(ubSize) / TYPE_SIZE), queueTensorCount), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAtanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AtanhCompileInfo {};

IMPL_OP_OPTILING(Atanh).Tiling(AtanhTilingFunc).TilingParse<AtanhCompileInfo>(TilingParseForAtanh);

} // namespace optiling
