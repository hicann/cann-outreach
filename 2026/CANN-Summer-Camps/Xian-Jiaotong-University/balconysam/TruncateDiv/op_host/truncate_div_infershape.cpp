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
 * \file truncate_div_infershape.cpp
 * \brief shape and dtype inference for TruncateDiv
 *        Implements standard NumPy-style broadcast shape derivation
 *        and dtype consistency check per the operator prototype.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include <vector>

using namespace ge;

namespace ops {

static constexpr int64_t MAX_SUPPORT_DIM = 8;

static ge::graphStatus InferShapeTruncateDiv(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE(context, "context is nullptr"), return GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin InferShapeTruncateDiv");

    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Use storage shapes for broadcast derivation
    const gert::StorageShape x1Storage = x1Shape->GetStorageShape();
    const gert::StorageShape x2Storage = x2Shape->GetStorageShape();

    int64_t rank1 = static_cast<int64_t>(x1Storage.GetDimNum());
    int64_t rank2 = static_cast<int64_t>(x2Storage.GetDimNum());
    int64_t outRank = (rank1 > rank2) ? rank1 : rank2;
    if (outRank > MAX_SUPPORT_DIM) {
        OP_LOGE(context, "input rank %lld exceeds max supported rank %lld",
                outRank, MAX_SUPPORT_DIM);
        return GRAPH_FAILED;
    }

    // Right-align the shapes and do broadcast.
    // For each output axis (left to right), the corresponding input dim
    // for a shorter shape is at offset (i - (outRank - thatRank)) when
    // that offset >= 0, otherwise the dim does not exist (treated as 1).
    std::vector<int64_t> outDims(outRank, 1);
    for (int64_t i = 0; i < outRank; ++i) {
        // Map output axis i to the right-aligned input dim
        int64_t x1Idx = i - (outRank - rank1);
        int64_t x2Idx = i - (outRank - rank2);
        int64_t d1 = (x1Idx >= 0) ? x1Storage.GetDim(x1Idx) : 1;
        int64_t d2 = (x2Idx >= 0) ? x2Storage.GetDim(x2Idx) : 1;

        // Unknown dim (-1) always passes broadcast check; output stays unknown
        if (d1 == -1 || d2 == -1) {
            outDims[i] = -1;
            continue;
        }

        if (d1 != d2 && d1 != 1 && d2 != 1) {
            OP_LOGE(context, "Incompatible broadcast dims: x1 dim %lld and x2 dim %lld at axis %lld",
                    d1, d2, i);
            return GRAPH_FAILED;
        }
        if (d1 == d2) {
            outDims[i] = d1;
        } else {
            outDims[i] = (d1 == 1) ? d2 : d1;
        }
    }

    yShape->SetDimNum(static_cast<uint32_t>(outRank));
    for (int64_t i = 0; i < outRank; ++i) {
        yShape->SetDim(static_cast<uint32_t>(i), outDims[i]);
    }

    OP_LOGD(context->GetNodeName(), "End InferShapeTruncateDiv, outRank=%lld", outRank);
    return GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  dtype derivation
// ---------------------------------------------------------------------------
// The output dtype is derived from the OpDef declaration in
// truncate_div_def.cpp: the output "y" declares the same DataType list as
// inputs "x1" and "x2", so the GE framework propagates input dtype to output
// automatically. No explicit InferDataType registration is needed.
// ---------------------------------------------------------------------------

IMPL_OP_INFERSHAPE(TruncateDiv)
    .InferShape(InferShapeTruncateDiv);

}  // namespace ops
