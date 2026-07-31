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
 * \file soft_shrink_grad_tiling.cpp
 * \brief SoftShrinkGrad host tiling implementation
 */

#include <set>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/soft_shrink_grad_tiling_data.h"
#include "../op_kernel/soft_shrink_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FLOAT_SIZE = 4;
constexpr int64_t HALF_SIZE = 2;
constexpr int64_t QUEUE_BUFFER_NUM = 2;
constexpr int64_t SELECT_TMP_RESERVED = 8 * 1024;
constexpr int64_t VECTOR_ALIGN_ELEMENTS = 64;

static int64_t GetShapeSize(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "SoftShrinkGrad: AIV core number is invalid"),
                return ge::GRAPH_FAILED);

    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "SoftShrinkGrad: UB size is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType,
                                    float& lambd)
{
    auto gradShape = context->GetInputShape(0);
    auto xShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    totalNum = GetShapeSize(gradShape->GetStorageShape());
    int64_t xNum = GetShapeSize(xShape->GetStorageShape());
    OP_CHECK_IF(totalNum != xNum,
                OP_LOGE(context, "SoftShrinkGrad: input element counts differ, grad=%ld, x=%ld", totalNum, xNum),
                return ge::GRAPH_FAILED);

    auto gradDesc = context->GetInputDesc(0);
    auto xDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    dataType = gradDesc->GetDataType();
    const std::set<ge::DataType> supportedTypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedTypes.count(dataType) == 0,
                OP_LOGE(context, "SoftShrinkGrad: unsupported input dtype=%d", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDesc->GetDataType() != dataType,
                OP_LOGE(context, "SoftShrinkGrad: input dtypes must be identical"), return ge::GRAPH_FAILED);

    lambd = 0.5f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* lambdAttr = attrs->GetAttrPointer<float>(0);
        if (lambdAttr != nullptr) {
            lambd = *lambdAttr;
        }
    }
    OP_CHECK_IF(lambd < 0.0f, OP_LOGE(context, "SoftShrinkGrad: lambd must be non-negative, got %f", lambd),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetBytesPerElement(ge::DataType dataType)
{
    int64_t typeSize = dataType == ge::DT_FLOAT ? FLOAT_SIZE : HALF_SIZE;
    int64_t computeBytes = 0;
    if (dataType == ge::DT_FLOAT16) {
        computeBytes = FLOAT_SIZE;
    } else if (dataType == ge::DT_BF16) {
        computeBytes = FLOAT_SIZE * 3;
    }
    return typeSize * QUEUE_BUFFER_NUM * 3 + computeBytes + 1;
}

static ge::graphStatus GetTilingKey(gert::TilingContext* context, ge::DataType dataType, uint64_t& tilingKey)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_0);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_1);
            return ge::GRAPH_SUCCESS;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_2);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "SoftShrinkGrad: unsupported dtype for tiling key");
            return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus SoftShrinkGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SoftShrinkGrad: failed to get platform information"),
                return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    float lambd = 0.5f;
    OP_CHECK_IF(GetInputInfo(context, totalNum, dataType, lambd) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SoftShrinkGrad: failed to get input information"),
                return ge::GRAPH_FAILED);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;

    auto tiling = context->GetTilingData<SoftShrinkGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SoftShrinkGradTilingData), 0, sizeof(SoftShrinkGradTilingData)) != EOK,
                OP_LOGE(context, "SoftShrinkGrad: failed to initialize tiling data"), return ge::GRAPH_FAILED);

    int64_t usableUb = static_cast<int64_t>(ubSize) - SELECT_TMP_RESERVED;
    OP_CHECK_IF(usableUb <= 0, OP_LOGE(context, "SoftShrinkGrad: insufficient UB space"),
                return ge::GRAPH_FAILED);
    int64_t bytesPerElement = GetBytesPerElement(dataType);
    int64_t ubFactor = FloorAlign(usableUb / bytesPerElement, VECTOR_ALIGN_ELEMENTS);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "SoftShrinkGrad: calculated UB factor is zero"),
                return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum > 0 ? CeilDiv(totalNum, coreNum) : 1;
    tiling->ubFactor = ubFactor;
    tiling->lambd = lambd;

    int64_t usedCoreNum = totalNum > 0 ? CeilDiv(totalNum, tiling->blockFactor) : 1;
    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey = 0;
    OP_CHECK_IF(GetTilingKey(context, dataType, tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SoftShrinkGrad: failed to select tiling key"),
                return ge::GRAPH_FAILED);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftShrinkGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftShrinkGradCompileInfo {};

IMPL_OP_OPTILING(SoftShrinkGrad)
    .Tiling(SoftShrinkGradTilingFunc)
    .TilingParse<SoftShrinkGradCompileInfo>(TilingParseForSoftShrinkGrad);

} // namespace optiling
