/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "truncate_div.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TruncateDiv);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32};

static bool CanBroadcast(const aclTensor* x1, const aclTensor* x2, op::Shape& broadcastShape)
{
    return BroadcastInferShape(x1->GetViewShape(), x2->GetViewShape(), broadcastShape);
}

static const aclTensor* TruncateDivAiCore(
    const aclTensor* x1, const aclTensor* x2,
    const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(TruncateDivAiCore, x1, x2);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(TruncateDiv, OP_INPUT(x1, x2), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncateDivAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* TruncateDiv(
    const aclTensor* x1, const aclTensor* x2,
    aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!CanBroadcast(x1, x2, broadcastShape)) {
        OP_LOGE(
            ACL_ERROR_INVALID_PARAM, "broadcast x1: %s, x2: %s failed.",
            op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, x1->GetDataType());
    return TruncateDivAiCore(x1, x2, out, executor);
}
} // namespace l0op
