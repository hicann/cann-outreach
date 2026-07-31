/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "truncatemod.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TruncateMod);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16,  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT32, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

// 根据 dtype 判断算子是否支持走 AiCore。
static bool IsAiCoreSupport(const aclTensor* self) { return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

// AICORE算子kernel
static const aclTensor* TruncateModAiCore(const aclTensor* self, const aclTensor* other,
                                          const aclTensor* truncateModOut, aclOpExecutor* executor)
{
    L0_DFX(TruncateModAiCore);
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCore TruncateMod算子加入任务队列
    // TruncateMod是算子的OpType，self、other是算子的输入，TruncateModOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(TruncateMod, OP_INPUT(self, other), OP_OUTPUT(truncateModOut));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncateModAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return truncateModOut;
}

const aclTensor* TruncateMod(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    // 根据输入shape推导输出shape
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
                op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    // 根据输出shape申请输出tensor
    if (!IsAiCoreSupport(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TruncateMod does not support dtype %s.",
                op::ToString(self->GetDataType()).GetString());
        return nullptr;
    }

    auto truncateModOut = executor->AllocTensor(broadcastShape, self->GetDataType(), op::Format::FORMAT_ND);
    OP_CHECK(truncateModOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for TruncateMod output failed."),
             return nullptr);
    return TruncateModAiCore(self, other, truncateModOut, executor);
}

} // namespace l0op
