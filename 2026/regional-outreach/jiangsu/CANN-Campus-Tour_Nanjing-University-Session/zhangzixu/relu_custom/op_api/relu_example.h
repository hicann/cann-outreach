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
 * \file relu_example.h
 * \brief ReluExample 算子 L0 API 头文件
 */

#ifndef ASCEND_CUSTOM_OP_RELU_EXAMPLE_L0_H
#define ASCEND_CUSTOM_OP_RELU_EXAMPLE_L0_H

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ReluExample 算子 L0 接口：获取 workspace 大小
 * @param [in] x: 输入 tensor
 * @param [in] y: 输出 tensor（预分配）
 * @param [out] workspaceSize: 返回的 workspace 大小
 * @param [out] executor: 返回的 executor 句柄
 * @return aclError
 */
aclError l0opReluExample(aclTensor* x, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

#ifdef __cplusplus
}
#endif
#endif
