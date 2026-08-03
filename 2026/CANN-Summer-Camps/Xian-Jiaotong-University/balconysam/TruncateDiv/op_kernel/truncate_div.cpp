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
 * \file truncate_div.cpp
 * \brief TruncateDiv Ascend C Kernel Implementation
 *
 * TruncateDiv: y = trunc(x1 / x2) - truncated division (round toward zero)
 *
 * Supported dtypes: fp16, fp32, bf16, int8, uint8, int32
 * Supported modes: same-shape, scalar broadcast, general broadcast
 */

#include "kernel_operator.h"
#include "truncate_div_tiling_data.h"

using namespace AscendC;

// ============================================================================
//  Constants
// ============================================================================
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

__aicore__ inline uint64_t GetBroadcastOffset(
    uint64_t outputOffset,
    const uint64_t inputStride[TRUNCATE_DIV_MAX_DIM],
    const uint64_t outputShape[TRUNCATE_DIV_MAX_DIM]) {
    uint64_t inputOffset = 0;
    for (int32_t dim = static_cast<int32_t>(TRUNCATE_DIV_MAX_DIM) - 1; dim >= 0; --dim) {
        uint64_t coordinate = outputOffset % outputShape[dim];
        outputOffset /= outputShape[dim];
        inputOffset += coordinate * inputStride[dim];
    }
    return inputOffset;
}

// ============================================================================
//  Float Type Kernel Template (fp16, fp32, bf16)
// ============================================================================
template <typename DTYPE>
class TruncateDivKernelFloat {
public:
    __aicore__ inline TruncateDivKernelFloat() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const TruncateDivTilingData* tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count);

private:
    GlobalTensor<DTYPE> x1Gm_, x2Gm_, yGm_;
    TQue<DTYPE, BUFFER_NUM> inQueueX1_, inQueueX2_, outQueueY_;
    TPipe pipe_;
    const TruncateDivTilingData* tiling_;
    uint64_t coreOffset_;
    uint64_t coreDataCount_;
};

template <typename DTYPE>
__aicore__ inline void TruncateDivKernelFloat<DTYPE>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    const TruncateDivTilingData* tiling) {

    tiling_ = tiling;
    uint64_t blockIdx = GetBlockIdx();
    coreOffset_ = blockIdx * tiling->coreLength;

    if (blockIdx < tiling->usedCoreNum - 1) {
        coreDataCount_ = tiling->coreLength;
    } else {
        coreDataCount_ = tiling->totalLength - coreOffset_;
        if (coreDataCount_ == 0) return;
    }

    x1Gm_.SetGlobalBuffer((__gm__ DTYPE*)x1, tiling->totalLength);
    x2Gm_.SetGlobalBuffer((__gm__ DTYPE*)x2, tiling->totalLength);
    yGm_.SetGlobalBuffer((__gm__ DTYPE*)y, tiling->totalLength);

    uint32_t tileBytes = ((tiling->tileLength * sizeof(DTYPE) + BLOCK_SIZE - 1)
                          / BLOCK_SIZE) * BLOCK_SIZE;
    pipe_.InitBuffer(inQueueX1_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(inQueueX2_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileBytes);
}

template <typename DTYPE>
__aicore__ inline void TruncateDivKernelFloat<DTYPE>::ProcessTile(
    uint64_t offset, uint32_t count) {

    LocalTensor<DTYPE> x1Local = inQueueX1_.AllocTensor<DTYPE>();
    LocalTensor<DTYPE> x2Local = inQueueX2_.AllocTensor<DTYPE>();
    LocalTensor<DTYPE> yLocal = outQueueY_.AllocTensor<DTYPE>();

    uint32_t bcastMode = tiling_->broadcastMode;

    if (bcastMode == 0) {
        DataCopy(x1Local, x1Gm_[offset], count);
        DataCopy(x2Local, x2Gm_[offset], count);
    } else if (bcastMode == 1) {
        DTYPE scalarX1 = x1Gm_.GetValue(0);
        Duplicate(x1Local, scalarX1, count);
        DataCopy(x2Local, x2Gm_[offset], count);
    } else if (bcastMode == 2) {
        DataCopy(x1Local, x1Gm_[offset], count);
        DTYPE scalarX2 = x2Gm_.GetValue(0);
        Duplicate(x2Local, scalarX2, count);
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            uint64_t outputOffset = offset + i;
            uint64_t x1Offset = GetBroadcastOffset(
                outputOffset, tiling_->x1Stride, tiling_->outputShape);
            uint64_t x2Offset = GetBroadcastOffset(
                outputOffset, tiling_->x2Stride, tiling_->outputShape);
            x1Local.SetValue(i, x1Gm_.GetValue(x1Offset));
            x2Local.SetValue(i, x2Gm_.GetValue(x2Offset));
        }
    }

    inQueueX1_.EnQue(x1Local);
    inQueueX2_.EnQue(x2Local);
    x1Local = inQueueX1_.DeQue<DTYPE>();
    x2Local = inQueueX2_.DeQue<DTYPE>();

    // Core: Div + Trunc
    Div(yLocal, x1Local, x2Local, count);
    Trunc(yLocal, yLocal, count);

    outQueueY_.EnQue<DTYPE>(yLocal);
    inQueueX1_.FreeTensor(x1Local);
    inQueueX2_.FreeTensor(x2Local);
    yLocal = outQueueY_.DeQue<DTYPE>();
    DataCopy(yGm_[offset], yLocal, count);
    outQueueY_.FreeTensor(yLocal);
}

template <typename DTYPE>
__aicore__ inline void TruncateDivKernelFloat<DTYPE>::Process() {

    if (coreDataCount_ == 0) return;

    uint64_t processed = 0;
    uint32_t tileLength = tiling_->tileLength;

    while (processed < coreDataCount_) {
        uint32_t currentTile = (coreDataCount_ - processed > tileLength)
                                 ? tileLength
                                 : static_cast<uint32_t>(coreDataCount_ - processed);

        ProcessTile(coreOffset_ + processed, currentTile);
        processed += currentTile;
    }
}

// ============================================================================
//  Int8/Uint8 Kernel (cast to half for computation)
// ============================================================================
template <typename DTYPE_INT>
class TruncateDivKernelInt8 {
public:
    __aicore__ inline TruncateDivKernelInt8() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const TruncateDivTilingData* tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count);

private:
    GlobalTensor<DTYPE_INT> x1Gm_, x2Gm_, yGm_;
    TQue<DTYPE_INT, BUFFER_NUM> inQueueX1_, inQueueX2_, outQueueY_;
    TBuf<half> halfBuf_;  // Buffer for half computation
    TPipe pipe_;
    const TruncateDivTilingData* tiling_;
    uint64_t coreOffset_;
    uint64_t coreDataCount_;
};

template <typename DTYPE_INT>
__aicore__ inline void TruncateDivKernelInt8<DTYPE_INT>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    const TruncateDivTilingData* tiling) {

    tiling_ = tiling;
    uint64_t blockIdx = GetBlockIdx();
    coreOffset_ = blockIdx * tiling->coreLength;

    if (blockIdx < tiling->usedCoreNum - 1) {
        coreDataCount_ = tiling->coreLength;
    } else {
        coreDataCount_ = tiling->totalLength - coreOffset_;
        if (coreDataCount_ == 0) return;
    }

    x1Gm_.SetGlobalBuffer((__gm__ DTYPE_INT*)x1, tiling->totalLength);
    x2Gm_.SetGlobalBuffer((__gm__ DTYPE_INT*)x2, tiling->totalLength);
    yGm_.SetGlobalBuffer((__gm__ DTYPE_INT*)y, tiling->totalLength);

    uint32_t tileBytes = tiling->tileLength * sizeof(DTYPE_INT);
    // Need 3 half buffers for x1_half, x2_half, y_half
    uint32_t halfTileBytes = tiling->tileLength * sizeof(half) * 3;

    pipe_.InitBuffer(inQueueX1_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(inQueueX2_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(halfBuf_, halfTileBytes);
}

template <typename DTYPE_INT>
__aicore__ inline void TruncateDivKernelInt8<DTYPE_INT>::ProcessTile(
    uint64_t offset, uint32_t count) {

    LocalTensor<DTYPE_INT> x1Int = inQueueX1_.AllocTensor<DTYPE_INT>();
    LocalTensor<DTYPE_INT> x2Int = inQueueX2_.AllocTensor<DTYPE_INT>();
    LocalTensor<DTYPE_INT> yInt = outQueueY_.AllocTensor<DTYPE_INT>();

    // Get half buffer for computation - single contiguous buffer for all half tensors
    LocalTensor<half> halfBase = halfBuf_.GetTensor<half>();
    LocalTensor<half> x1Half = halfBase;                    // offset 0
    LocalTensor<half> x2Half = halfBase[count];             // offset after x1
    LocalTensor<half> yHalf = halfBase[count * 2];          // offset after x2

    uint32_t bcastMode = tiling_->broadcastMode;
    if (bcastMode == 0) {
        DataCopy(x1Int, x1Gm_[offset], count);
        DataCopy(x2Int, x2Gm_[offset], count);
    } else if (bcastMode == 1) {
        DTYPE_INT scalarX1 = x1Gm_.GetValue(0);
        Duplicate(x1Int, scalarX1, count);
        DataCopy(x2Int, x2Gm_[offset], count);
    } else if (bcastMode == 2) {
        DataCopy(x1Int, x1Gm_[offset], count);
        DTYPE_INT scalarX2 = x2Gm_.GetValue(0);
        Duplicate(x2Int, scalarX2, count);
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            uint64_t outputOffset = offset + i;
            uint64_t x1Offset = GetBroadcastOffset(
                outputOffset, tiling_->x1Stride, tiling_->outputShape);
            uint64_t x2Offset = GetBroadcastOffset(
                outputOffset, tiling_->x2Stride, tiling_->outputShape);
            x1Int.SetValue(i, x1Gm_.GetValue(x1Offset));
            x2Int.SetValue(i, x2Gm_.GetValue(x2Offset));
        }
    }

    inQueueX1_.EnQue(x1Int);
    inQueueX2_.EnQue(x2Int);
    x1Int = inQueueX1_.DeQue<DTYPE_INT>();
    x2Int = inQueueX2_.DeQue<DTYPE_INT>();

    // Cast to half for computation
    Cast(x1Half, x1Int, RoundMode::CAST_NONE, count);
    Cast(x2Half, x2Int, RoundMode::CAST_NONE, count);

    // Half division and truncation
    Div(yHalf, x1Half, x2Half, count);
    Trunc(yHalf, yHalf, count);

    // Cast back to int8/uint8
    Cast(yInt, yHalf, RoundMode::CAST_RINT, count);

    outQueueY_.EnQue<DTYPE_INT>(yInt);
    inQueueX1_.FreeTensor(x1Int);
    inQueueX2_.FreeTensor(x2Int);
    yInt = outQueueY_.DeQue<DTYPE_INT>();
    DataCopy(yGm_[offset], yInt, count);
    outQueueY_.FreeTensor(yInt);
}

template <typename DTYPE_INT>
__aicore__ inline void TruncateDivKernelInt8<DTYPE_INT>::Process() {

    if (coreDataCount_ == 0) return;

    uint64_t processed = 0;
    uint32_t tileLength = tiling_->tileLength;

    while (processed < coreDataCount_) {
        uint32_t count = (coreDataCount_ - processed > tileLength)
                           ? tileLength
                           : static_cast<uint32_t>(coreDataCount_ - processed);
        ProcessTile(coreOffset_ + processed, count);
        processed += count;
    }
}

// ============================================================================
//  Int32 Kernel (cast to float for computation)
// ============================================================================
class TruncateDivKernelInt32 {
public:
    __aicore__ inline TruncateDivKernelInt32() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const TruncateDivTilingData* tiling) {
        tiling_ = tiling;
        uint64_t blockIdx = GetBlockIdx();
        coreOffset_ = blockIdx * tiling->coreLength;

        if (blockIdx < tiling->usedCoreNum - 1) {
            coreDataCount_ = tiling->coreLength;
        } else {
            coreDataCount_ = tiling->totalLength - coreOffset_;
            if (coreDataCount_ == 0) return;
        }

        x1Gm_.SetGlobalBuffer((__gm__ int32_t*)x1, tiling->totalLength);
        x2Gm_.SetGlobalBuffer((__gm__ int32_t*)x2, tiling->totalLength);
        yGm_.SetGlobalBuffer((__gm__ int32_t*)y, tiling->totalLength);

        uint32_t tileBytes = tiling->tileLength * sizeof(int32_t);
        // Need 3 float buffers for x1_float, x2_float, y_float
        uint32_t floatTileBytes = tiling->tileLength * sizeof(float) * 3;

        pipe_.InitBuffer(inQueueX1_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(inQueueX2_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(floatBuf_, floatTileBytes);
    }

    __aicore__ inline void Process() {
        if (coreDataCount_ == 0) return;

        uint64_t processed = 0;
        uint32_t tileLength = tiling_->tileLength;

        while (processed < coreDataCount_) {
            uint32_t count = (coreDataCount_ - processed > tileLength)
                               ? tileLength
                               : static_cast<uint32_t>(coreDataCount_ - processed);
            ProcessTile(coreOffset_ + processed, count);
            processed += count;
        }
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count) {
        LocalTensor<int32_t> x1Int = inQueueX1_.AllocTensor<int32_t>();
        LocalTensor<int32_t> x2Int = inQueueX2_.AllocTensor<int32_t>();
        LocalTensor<int32_t> yInt = outQueueY_.AllocTensor<int32_t>();

        // Single contiguous buffer for all float tensors
        LocalTensor<float> floatBase = floatBuf_.GetTensor<float>();
        LocalTensor<float> x1Float = floatBase;               // offset 0
        LocalTensor<float> x2Float = floatBase[count];        // offset after x1
        LocalTensor<float> yFloat = floatBase[count * 2];     // offset after x2

        uint32_t bcastMode = tiling_->broadcastMode;
        if (bcastMode == 0) {
            DataCopy(x1Int, x1Gm_[offset], count);
            DataCopy(x2Int, x2Gm_[offset], count);
        } else if (bcastMode == 1) {
            int32_t scalarX1 = x1Gm_.GetValue(0);
            Duplicate(x1Int, scalarX1, count);
            DataCopy(x2Int, x2Gm_[offset], count);
        } else if (bcastMode == 2) {
            DataCopy(x1Int, x1Gm_[offset], count);
            int32_t scalarX2 = x2Gm_.GetValue(0);
            Duplicate(x2Int, scalarX2, count);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                uint64_t outputOffset = offset + i;
                uint64_t x1Offset = GetBroadcastOffset(
                    outputOffset, tiling_->x1Stride, tiling_->outputShape);
                uint64_t x2Offset = GetBroadcastOffset(
                    outputOffset, tiling_->x2Stride, tiling_->outputShape);
                x1Int.SetValue(i, x1Gm_.GetValue(x1Offset));
                x2Int.SetValue(i, x2Gm_.GetValue(x2Offset));
            }
        }

        inQueueX1_.EnQue(x1Int);
        inQueueX2_.EnQue(x2Int);
        x1Int = inQueueX1_.DeQue<int32_t>();
        x2Int = inQueueX2_.DeQue<int32_t>();

        // Cast to float
        Cast(x1Float, x1Int, RoundMode::CAST_NONE, count);
        Cast(x2Float, x2Int, RoundMode::CAST_NONE, count);

        // Float division and truncation
        Div(yFloat, x1Float, x2Float, count);
        Trunc(yFloat, yFloat, count);

        // Cast back to int32
        Cast(yInt, yFloat, RoundMode::CAST_RINT, count);

        outQueueY_.EnQue<int32_t>(yInt);
        inQueueX1_.FreeTensor(x1Int);
        inQueueX2_.FreeTensor(x2Int);
        yInt = outQueueY_.DeQue<int32_t>();
        DataCopy(yGm_[offset], yInt, count);
        outQueueY_.FreeTensor(yInt);
    }

private:
    GlobalTensor<int32_t> x1Gm_, x2Gm_, yGm_;
    TQue<int32_t, BUFFER_NUM> inQueueX1_, inQueueX2_, outQueueY_;
    TBuf<float> floatBuf_;
    TPipe pipe_;
    const TruncateDivTilingData* tiling_;
    uint64_t coreOffset_, coreDataCount_;
};

// ============================================================================
//  Kernel Entry Point
// ============================================================================
extern "C" __global__ __aicore__ void truncate_div(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tilingParam) {

    GM_ADDR tilingPtr = tilingParam;
    REGISTER_TILING_DEFAULT(TruncateDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateDivTilingData, tilingData, tilingPtr);

    uint32_t dtypeMode = tilingData.dtypeMode;
    // Standard computation paths
    if (dtypeMode == 0) {  // fp16
        TruncateDivKernelFloat<half> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if (dtypeMode == 1) {  // fp32
        TruncateDivKernelFloat<float> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if (dtypeMode == 2) {  // bf16
        TruncateDivKernelFloat<bfloat16> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if (dtypeMode == 3) {  // int8
        TruncateDivKernelInt8<int8_t> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if (dtypeMode == 4) {  // uint8
        TruncateDivKernelInt8<uint8_t> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else {  // int32
        TruncateDivKernelInt32 op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
}
