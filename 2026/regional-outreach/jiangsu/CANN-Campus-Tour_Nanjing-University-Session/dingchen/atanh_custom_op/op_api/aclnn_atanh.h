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
 * @file aclnn_atanh.h
 * @brief ACLNN L2 API 接口声明 - Atanh 一元算子
 *
 * ACLNN 接口采用两段式设计：
 * - GetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnAtanh: 执行计算
 */

#ifndef ACLNN_ATANH_H_
#define ACLNN_ATANH_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAtanh 所需的 workspace 大小
 * @param x [in] 输入张量
 * @param out [in] 输出张量
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAtanhGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 Atanh 算子计算
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAtanh(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ATANH_H_
