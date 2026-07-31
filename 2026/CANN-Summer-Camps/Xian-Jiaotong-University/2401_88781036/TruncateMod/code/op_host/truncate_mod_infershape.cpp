/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t INPUT_X1 = 0;
static constexpr int64_t INPUT_X2 = 1;
static constexpr int64_t OUTPUT_Y = 0;

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    const gert::Shape* x1Shape = context->GetInputShape(INPUT_X1);
    const gert::Shape* x2Shape = context->GetInputShape(INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const size_t x1Rank = x1Shape->GetDimNum();
    const size_t x2Rank = x2Shape->GetDimNum();
    const size_t outRank = x1Rank > x2Rank ? x1Rank : x2Rank;

    gert::Shape outShape;
    for (size_t i = 0; i < outRank; ++i) {
        int64_t dim1 = 1;
        int64_t dim2 = 1;

        if (i >= outRank - x1Rank) {
            dim1 = x1Shape->GetDim(i - (outRank - x1Rank));
        }
        if (i >= outRank - x2Rank) {
            dim2 = x2Shape->GetDim(i - (outRank - x2Rank));
        }

        int64_t outDim = 1;
        if (dim1 == dim2) {
            outDim = dim1;
        } else if (dim1 == 1) {
            outDim = dim2;
        } else if (dim2 == 1) {
            outDim = dim1;
        } else if (dim1 < 0 || dim2 < 0) {
            outDim = -1;
        } else {
            OP_LOGE(context, "TruncateMod input shapes cannot broadcast");
            return ge::GRAPH_FAILED;
        }

        outShape.AppendDim(outDim);
    }

    *yShape = outShape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod);
} // namespace ops
