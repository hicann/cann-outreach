/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod 算子 Tiling 实现
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "truncate_mod_tiling.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"
#include <limits>

namespace optiling {
static constexpr uint32_t BLOCK_SIZE = 32U;
static constexpr uint32_t MAX_CORE_NUM = 40U;
// Large contiguous/scalar-broadcast tensors are bandwidth bound.  A larger
// tile amortizes queue, vector-instruction and tail handling overhead while
// remaining comfortably below the Ascend910B 192 KiB UB budget for every
// supported dtype.  General broadcasting keeps a smaller tile because its
// gather path is register/index bound.
static constexpr uint32_t CONTIGUOUS_TILE_DATA_NUM = 4096U;
static constexpr uint32_t BROADCAST_TILE_DATA_NUM = 1024U;
static constexpr uint32_t CORE_GRAIN = 128U;

static uint32_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

static ge::graphStatus GetTotalLength(gert::TilingContext* context, uint32_t& totalLength)
{
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    int64_t totalLength64 = outputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalLength64 <= 0 || totalLength64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context, "invalid total length"), return ge::GRAPH_FAILED);

    totalLength = static_cast<uint32_t>(totalLength64);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillShape(const gert::Shape& shape, uint32_t* dims)
{
    const size_t rank = shape.GetDimNum();
    if (rank > TRUNCATE_MOD_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < TRUNCATE_MOD_MAX_DIMS; ++i) {
        dims[i] = 1U;
    }

    const size_t leading = TRUNCATE_MOD_MAX_DIMS - rank;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return ge::GRAPH_FAILED;
        }
        dims[leading + i] = static_cast<uint32_t>(dim);
    }
    return ge::GRAPH_SUCCESS;
}

struct TilingParams {
    uint32_t tileDataNum;
    uint32_t blockDim;
    uint32_t blockLength;
    uint32_t tailDataNum;
};

static ge::graphStatus PrepareShapeInfo(gert::TilingContext* context, TruncateModTilingData* tiling,
                                        uint32_t& x1DataNum, uint32_t& x2DataNum)
{
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    const gert::StorageShape* x2Shape = context->GetInputShape(1);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const gert::Shape& x1StorageShape = x1Shape->GetStorageShape();
    const gert::Shape& x2StorageShape = x2Shape->GetStorageShape();
    const gert::Shape& yStorageShape = outputShape->GetStorageShape();

    const int64_t x1DataNum64 = x1StorageShape.GetShapeSize();
    const int64_t x2DataNum64 = x2StorageShape.GetShapeSize();
    OP_CHECK_IF(x1DataNum64 <= 0 || x2DataNum64 <= 0 ||
                    x1DataNum64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
                    x2DataNum64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context, "invalid input size"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(FillShape(x1StorageShape, tiling->x1Shape) != ge::GRAPH_SUCCESS ||
                    FillShape(x2StorageShape, tiling->x2Shape) != ge::GRAPH_SUCCESS ||
                    FillShape(yStorageShape, tiling->outShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "rank exceeds limit or shape is invalid"), return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < TRUNCATE_MOD_MAX_DIMS; ++i) {
        OP_CHECK_IF((tiling->x1Shape[i] != 1U && tiling->x1Shape[i] != tiling->outShape[i]) ||
                        (tiling->x2Shape[i] != 1U && tiling->x2Shape[i] != tiling->outShape[i]),
                    OP_LOGE(context, "input shapes cannot broadcast to output"), return ge::GRAPH_FAILED);
    }

    x1DataNum = static_cast<uint32_t>(x1DataNum64);
    x2DataNum = static_cast<uint32_t>(x2DataNum64);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTilingParams(gert::TilingContext* context, uint32_t totalLength, uint32_t x1DataNum,
                                             uint32_t x2DataNum, TilingParams& params)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    uint32_t typeSize = GetTypeSize(inputDesc->GetDataType());
    if (typeSize == 0U) {
        OP_LOGE(context, "unsupported input data type");
        return ge::GRAPH_FAILED;
    }

    const bool hasGeneralBroadcast = (x1DataNum != totalLength && x1DataNum != 1U) ||
                                     (x2DataNum != totalLength && x2DataNum != 1U);
    params.tileDataNum = hasGeneralBroadcast ? BROADCAST_TILE_DATA_NUM : CONTIGUOUS_TILE_DATA_NUM;
    OP_CHECK_IF(params.tileDataNum == 0U, OP_LOGE(context, "tileDataNum is 0"), return ge::GRAPH_FAILED);

    uint32_t blockAlign = BLOCK_SIZE / typeSize;
    if (blockAlign == 0U) {
        blockAlign = 1U;
    }

    uint32_t usefulCoreNum = (totalLength + CORE_GRAIN - 1U) / CORE_GRAIN;
    params.blockDim = usefulCoreNum < MAX_CORE_NUM ? usefulCoreNum : MAX_CORE_NUM;
    if (params.blockDim == 0U) {
        params.blockDim = 1U;
    }

    params.blockLength = (totalLength + params.blockDim - 1U) / params.blockDim;
    params.blockLength = ((params.blockLength + blockAlign - 1U) / blockAlign) * blockAlign;
    params.blockDim = (totalLength + params.blockLength - 1U) / params.blockLength;
    params.tailDataNum = params.blockLength % params.tileDataNum;
    if (params.tailDataNum == 0U) {
        params.tailDataNum = params.tileDataNum;
    }
    return ge::GRAPH_SUCCESS;
}

static void FillTilingData(TruncateModTilingData* tiling, uint32_t totalLength, uint32_t x1DataNum, uint32_t x2DataNum,
                           const TilingParams& params)
{
    uint32_t finalTileNum = (params.blockLength + params.tileDataNum - 1U) / params.tileDataNum;
    tiling->smallCoreDataNum = params.blockLength;
    tiling->bigCoreDataNum = params.blockLength;
    tiling->finalSmallTileNum = finalTileNum;
    tiling->finalBigTileNum = finalTileNum;
    tiling->smallTailDataNum = params.tailDataNum;
    tiling->bigTailDataNum = params.tailDataNum;
    tiling->tailBlockNum = params.blockDim;
    tiling->totalDataNum = totalLength;
    tiling->tmpTileDataNum = params.tileDataNum * 2U;
    tiling->tmpSmallTailDataNum = params.tailDataNum * 2U;
    tiling->tmpBigTailDataNum = params.tailDataNum * 2U;
    tiling->x1DataNum = x1DataNum;
    tiling->x2DataNum = x2DataNum;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint32_t totalLength = 0;
    OP_CHECK_IF(GetTotalLength(context, totalLength) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetTotalLength failed"),
                return ge::GRAPH_FAILED);

    uint32_t x1DataNum = 0;
    uint32_t x2DataNum = 0;
    OP_CHECK_IF(PrepareShapeInfo(context, tiling, x1DataNum, x2DataNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "PrepareShapeInfo failed"), return ge::GRAPH_FAILED);

    TilingParams params{};
    OP_CHECK_IF(CalculateTilingParams(context, totalLength, x1DataNum, x2DataNum, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CalculateTilingParams failed"), return ge::GRAPH_FAILED);
    FillTilingData(tiling, totalLength, x1DataNum, x2DataNum, params);

    context->SetBlockDim(params.blockDim);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateMod(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateMod).Tiling(TilingFunc).TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);
} // namespace optiling
