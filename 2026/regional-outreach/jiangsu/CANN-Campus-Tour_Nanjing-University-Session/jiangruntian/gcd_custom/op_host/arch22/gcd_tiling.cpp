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
 * \file gcd_tiling.cpp
 * \brief GCD 算子 Tiling 实现（arch22 - ascend910b）
 *
 * Tiling 策略：
 *   1. 获取总元素数（broadcast 后的总元素数）
 *   2. 按可用核数均分（多核并行）
 *   3. 按 UB 大小切分 tile（流水线）
 *   4. 根据数据量选择单缓冲/双缓冲
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/gcd_tiling_data.h"
#include "../../op_kernel/arch22/gcd_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 2;  // float16 = 2 bytes
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 3;
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 6;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShapeSelf = EnsureNotScalar(inputSelf->GetStorageShape());

    auto inputOther = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputOther);
    auto inputShapeOther = EnsureNotScalar(inputOther->GetStorageShape());

    auto out = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);
    auto outShape = EnsureNotScalar(out->GetStorageShape());

    // 校验广播后输入与输出元素数一致
    int64_t selfSize = inputShapeSelf.GetShapeSize();
    int64_t otherSize = inputShapeOther.GetShapeSize();
    int64_t outSize = outShape.GetShapeSize();
    OP_CHECK_IF(outSize != selfSize && outSize != otherSize && selfSize != otherSize,
        OP_LOGE(context, "Gcd tiling: broadcast shapes must converge: self=%ld, other=%ld, out=%ld",
                selfSize, otherSize, outSize),
        return ge::GRAPH_FAILED);

    *totalIdx = outSize;

    // dtype 校验
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(*dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Gcd: only float16 supported, got %d.", static_cast<int>(*dataType)),
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

static ge::graphStatus GcdTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取 shape 信息
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 获取 WorkspaceSize
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. 设置 TilingData
    GcdTilingData* tiling = context->GetTilingData<GcdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(GcdTilingData), 0, sizeof(GcdTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分
    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB 切分
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    tiling->ubFactor = FloorAlign(FloorDiv((static_cast<int64_t>(ubSize) / TYPE_SIZE), bufferNum), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5. 设置 TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGcd([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GcdCompileInfo {};

IMPL_OP_OPTILING(Gcd).Tiling(GcdTilingFunc).TilingParse<GcdCompileInfo>(TilingParseForGcd);

} // namespace optiling
