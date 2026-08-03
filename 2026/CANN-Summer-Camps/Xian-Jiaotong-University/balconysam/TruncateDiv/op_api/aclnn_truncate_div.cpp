/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_truncate_div.h"
#include "truncate_div.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SUPPORT_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32};

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);

    if (x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x1 dtype %s and x2 dtype %s must be the same.",
            op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() != x1->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "out dtype %s must be the same as input dtype %s.",
            op::ToString(out->GetDataType()).GetString(),
            op::ToString(x1->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckMaxDimension(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x1, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(x2, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIM, return false);

    return true;
}

static bool CheckInAndOutShape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(x1, x2, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x1, x2, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckMaxDimension(x1, x2, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInAndOutShape(x1, x2, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTruncateDivGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnTruncateDiv, DFX_IN(x1, x2), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x1->IsEmpty() || x2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto truncateDivOut = l0op::TruncateDiv(x1Contiguous, x2Contiguous, uniqueExecutor.get());
    CHECK_RET(truncateDivOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(truncateDivOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTruncateDiv(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTruncateDiv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
