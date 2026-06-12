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
 * \file relu_example.cpp
 * \brief ReluExample 算子 L0 API 实现
 *
 * L0 API 核心流程：InferShape → CheckParams → AllocTensor → ReluExampleAiCore
 */

#include "relu_example.h"
#include "op_common/op_log.h"

namespace l0op {

struct ReluExampleL0Params {
    aclTensor* x;
    aclTensor* y;
};

// Shape 推导：输出 shape = 输入 shape
static aclError InferShape(const ReluExampleL0Params& params)
{
    auto inputShape = aclGetTensorShape(params.x);
    auto ndim = aclGetTensorDimNum(params.x);

    // 设置输出 shape 与输入一致
    for (size_t i = 0; i < ndim; i++) {
        aclSetTensorShape(params.y, i, inputShape[i]);
    }
    return ACL_SUCCESS;
}

// 参数校验
static aclError CheckParams(const ReluExampleL0Params& params)
{
    // 数据类型校验
    aclDataType xType = aclGetTensorDataType(params.x);
    aclDataType yType = aclGetTensorDataType(params.y);
    if (xType != ACL_FLOAT16 || yType != ACL_FLOAT16) {
        ASCEND_LOG_ERROR("ReluExample: only float16 supported");
        return ACL_ERROR_INVALID_PARAM;
    }
    // Shape 校验：输入输出 shape 一致
    if (aclGetTensorElementNum(params.x) != aclGetTensorElementNum(params.y)) {
        ASCEND_LOG_ERROR("ReluExample: input and output element count mismatch");
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

} // namespace l0op

aclError l0opReluExample(aclTensor* x, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    l0op::ReluExampleL0Params params = {x, y};

    // 1. Shape 推导
    l0op::InferShape(params);

    // 2. 参数校验
    l0op::CheckParams(params);

    // 3. 创建 executor（内部调用算子 kernel）
    // 注：实际由框架根据算子定义下发
    *workspaceSize = 0;
    *executor = nullptr;
    return ACL_SUCCESS;
}
