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
 * \file aclnn_relu_example.cpp
 * \brief ReluExample 算子 L2 API 实现
 *
 * L2 API 两段式调用模式：
 *   1. aclnnReluExampleGetWorkspaceSize — CREATE_EXECUTOR → CheckParams → l0opReluExample
 *   2. aclnnReluExample — 执行算子
 */

#include "aclnn_relu_example.h"
#include "relu_example.h"

// 第一段接口：获取 workspace 大小
aclError aclnnReluExampleGetWorkspaceSize(aclTensor* x, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return l0opReluExample(x, y, workspaceSize, executor);
}

// 第二段接口：执行算子
aclError aclnnReluExample(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
{
    // 注：实际执行由框架根据 executor 下发 kernel
    // 此处为占位，完整实现参考 relu_example 工程中的 op_api
    (void)workspace;
    (void)workspaceSize;
    (void)executor;
    (void)stream;
    return ACL_SUCCESS;
}
