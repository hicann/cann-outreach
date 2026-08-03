/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_TRUNCATE_DIV_H_
#define OP_API_INC_TRUNCATE_DIV_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTruncateDiv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对x1和x2执行逐元素向零取整除法（Truncated Division）。
 * 计算公式：
 * $$ y_i = trunc(x1_i / x2_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *   A[(x1)] -->B([l0op::Contiguous])
 *   B --> C([l0op::TruncateDiv])
 *   D[(x2)] -->E([l0op::Contiguous])
 *   E --> C
 *   C --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 *
 * @param [in] x1: npu
 * 被除数张量，npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32，
 * 数据类型需与x2保持一致，shape需与x2满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] x2: npu
 * 除数张量，npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32，
 * 数据类型需与x1保持一致，shape需与x1满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * 输出张量，npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32，
 * 数据类型需与x1、x2一致，shape是x1、x2 broadcast之后的shape，支持非连续Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTruncateDivGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnTruncateDiv的第二段接口，用于执行计算。
 *
 * 算子功能：对x1和x2执行逐元素向零取整除法（Truncated Division）。
 * 计算公式：
 * $$ y_i = trunc(x1_i / x2_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *   A[(x1)] -->B([l0op::Contiguous])
 *   B --> C([l0op::TruncateDiv])
 *   D[(x2)] -->E([l0op::Contiguous])
 *   E --> C
 *   C --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnTruncateDivGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnTruncateDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_TRUNCATE_DIV_H_
