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
 * @file gcd.cpp
 * @brief ACLNN L0 API 实现 - 二元最大公约数算子（float16，支持 broadcast）
 *
 * L0 API 职责：形状推导（broadcast）、Kernel 调度
 * L2 API 职责：参数检查、Contiguous/ViewCopy 处理
 */

#include "gcd.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Gcd);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16
};

static bool IsAiCoreSupport(const aclTensor* self, const aclTensor* other)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(other->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Gcd not supported: self_dtype=%d, other_dtype=%d. Supported: FLOAT16.",
                     static_cast<int>(self->GetDataType()), static_cast<int>(other->GetDataType())),
             return false);
    return true;
}

static bool GcdInferShape(const op::Shape& selfShape, const op::Shape& otherShape,
                           op::Shape& outShape)
{
    // 标准 broadcast 规则：尾部对齐，逐维度取 max
    OP_CHECK(BroadcastInferShape(selfShape, otherShape, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd: shape broadcast failed."), return false);
    return true;
}

static const aclTensor* GcdAiCore(const aclTensor* self, const aclTensor* other,
                                   const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(GcdAiCore, self, other, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Gcd,
        OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GcdAiCore failed."),
             return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 形状推导（broadcast）
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor     - 分配输出 Tensor
 * 4. GcdAiCore       - 调用 Kernel
 */
const aclTensor* Gcd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(GcdInferShape(self->GetViewShape(), other->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, self->GetDataType());

    OP_CHECK(IsAiCoreSupport(self, other),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return GcdAiCore(self, other, out, executor);
}

} // namespace l0op
