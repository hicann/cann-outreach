/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_TRUNCATE_MOD_TENSOR_H_
#define OP_API_INC_ACLNN_TRUNCATE_MOD_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "../../../../common/inc/external/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取 TruncateMod Tensor-Out 接口所需的 workspace 和执行器。
 * @domain aclnn_math
 *
 * 逐元素计算：out = self - trunc(self / other) * other，其中 trunc 向零截断。
 * self 与 other 支持广播，最大维数为 8。支持 BF16、FLOAT16、FLOAT、INT32、INT8、UINT8。
 *
 * @param [in] self 输入 Tensor，支持非连续 Tensor 和 ND 格式。
 * @param [in] other 输入 Tensor，需与 self 满足广播关系。
 * @param [in,out] out 输出 Tensor，shape 为两个输入广播后的 shape。
 * @param [out] workspaceSize Device workspace 大小。
 * @param [out] executor 算子执行器。
 * @return aclnnStatus 执行状态。
 */
ACLNN_API aclnnStatus aclnnTruncateModTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* other,
                                                             aclTensor* out, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief 执行 TruncateMod Tensor-Out 计算。
 * @param [in] workspace Device workspace 地址。
 * @param [in] workspaceSize 第一段接口返回的 workspace 大小。
 * @param [in] executor 第一段接口返回的执行器。
 * @param [in] stream ACL stream。
 * @return aclnnStatus 执行状态。
 */
ACLNN_API aclnnStatus aclnnTruncateModTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

/**
 * @brief 获取 TruncateMod 原地接口所需的 workspace 和执行器。
 * @param [in,out] selfRef 输入及输出 Tensor。
 * @param [in] other 输入 Tensor，需与 selfRef 满足广播关系且结果 shape 与 selfRef 相同。
 * @param [out] workspaceSize Device workspace 大小。
 * @param [out] executor 算子执行器。
 * @return aclnnStatus 执行状态。
 */
ACLNN_API aclnnStatus aclnnInplaceTruncateModTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* other,
                                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 TruncateMod 原地计算。
 * @param [in] workspace Device workspace 地址。
 * @param [in] workspaceSize 第一段接口返回的 workspace 大小。
 * @param [in] executor 第一段接口返回的执行器。
 * @param [in] stream ACL stream。
 * @return aclnnStatus 执行状态。
 */
ACLNN_API aclnnStatus aclnnInplaceTruncateModTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_TRUNCATE_MOD_TENSOR_H_
