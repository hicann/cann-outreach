/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file atanh.cpp
 * @brief ACLNN L0 API 实现 - Atanh 一元算子
 *
 * L0 API 职责：形状推导、Kernel 调度
 * 计算公式: atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 */

#include "atanh.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Atanh);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Atanh not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Atanh not supported: dtype x=%d. Supported dtype: FLOAT16.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool AtanhInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    outShape = xShape;
    return true;
}

static const aclTensor* AtanhAiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AtanhAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Atanh,
        OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AtanhAiCore failed."),
        return nullptr);
    return out;
}

const aclTensor* Atanh(const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;

    OP_CHECK(AtanhInferShape(x->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, x->GetDataType());

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return AtanhAiCore(x, out, executor);
}

} // namespace l0op
