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
 * \file gcd_infershape.cpp
 * \brief GCD 算子形状推导实现
 *
 * 广播规则（与 NumPy 语义一致）：
 *   1. 从右向左对齐维度
 *   2. 每个维度上，size=1 会广播到另一侧的对应 size
 *   3. 若某维度在一侧不存在，等效为 size=1
 *   4. 若两个 size 都不为 1 且不相等，报错
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Gcd(gert::InferShapeContext* context)
{
    // 获取输入 shape
    const gert::Shape* selfShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShape);

    const gert::Shape* otherShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, otherShape);

    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    // 广播推导：从右向左对齐
    auto selfDim = selfShape->GetDimNum();
    auto otherDim = otherShape->GetDimNum();
    size_t maxDim = (selfDim > otherDim) ? selfDim : otherDim;

    std::vector<int64_t> broadcastDims(maxDim);

    for (size_t i = 0; i < maxDim; i++) {
        int64_t dSelf = 1;
        int64_t dOther = 1;

        if (i < selfDim) {
            dSelf = selfShape->GetDim(selfDim - 1 - i);
        }
        if (i < otherDim) {
            dOther = otherShape->GetDim(otherDim - 1 - i);
        }

        // 广播检查
        if (dSelf != dOther && dSelf != 1 && dOther != 1) {
            OP_LOGE(context,
                    "Gcd: shape mismatch at dim %zu from right: self=%ld, other=%ld. "
                    "Broadcast requires equal dims or one dim to be 1.",
                    i, dSelf, dOther);
            return ge::GRAPH_FAILED;
        }

        broadcastDims[maxDim - 1 - i] = (dSelf > dOther) ? dSelf : dOther;
    }

    // 设置输出 shape
    outShape->SetDimNum(maxDim);
    for (size_t i = 0; i < maxDim; i++) {
        if (outShape->SetDim(i, broadcastDims[i]) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context, "Gcd: failed to set output dim %zu = %ld.", i, broadcastDims[i]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Gcd).InferShape(InferShape4Gcd);

} // namespace ops
