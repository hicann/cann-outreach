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
 * \file aclnn_relu_example.h
 * \brief ReluExample 算子 L2 API 头文件
 *
 * L2 API 对外暴露两段式接口：
 *   aclnnReluExampleGetWorkspaceSize — 第一段：获取 workspace 大小
 *   aclnnReluExample — 第二段：执行算子
 */

#ifndef ASCEND_CUSTOM_OP_ACLNN_RELU_EXAMPLE_H
#define ASCEND_CUSTOM_OP_ACLNN_RELU_EXAMPLE_H

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ReluExample 第一段接口：获取 workspace 大小
 * @param [in] x: 输入 tensor
 * @param [in] y: 输出 tensor（预分配）
 * @param [out] workspaceSize: 返回的 workspace 大小
 * @param [out] executor: 返回的 executor 句柄
 * @return aclError
 */
aclError aclnnReluExampleGetWorkspaceSize(aclTensor* x, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief ReluExample 第二段接口：执行算子
 * @param [in] workspace: workspace 地址
 * @param [in] workspaceSize: workspace 大小
 * @param [in] executor: executor 句柄
 * @param [in] stream: acl 流
 * @return aclError
 */
aclError aclnnReluExample(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
