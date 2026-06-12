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
 * @file aclnn_atanh.cpp
 * @brief ACLNN L2 API 实现 - Atanh 一元算子
 *
 * ACLNN 接口采用两段式设计：
 * 1. aclnnAtanhGetWorkspaceSize - 计算 workspace 大小，创建执行器
 * 2. aclnnAtanh - 执行计算
 */

#include "aclnn_atanh.h"
#include "atanh.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* x)
{
    return x->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: FLOAT16.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* out)
{
    auto formatX = x->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(formatX) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: x=%d, out=%d",
                     static_cast<int>(formatX), static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x_dtype=%d, out_dtype=%d",
               static_cast<int>(x->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(x, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: x_format=%d, out_format=%d",
               static_cast<int>(x->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: x_dim=%zu, out_dim=%zu",
               x->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAtanhGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAtanh, DFX_IN(x), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Atanh(xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAtanh(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAtanh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
