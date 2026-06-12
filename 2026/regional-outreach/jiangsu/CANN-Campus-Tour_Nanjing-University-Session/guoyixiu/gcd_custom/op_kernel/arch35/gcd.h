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
 * \file gcd.h
 * \brief Gcd 算子 kernel 类定义（arch35，二元 broadcast，float16）
 */
#ifndef GCD_H
#define GCD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gcd_tiling_data.h"
#include "gcd_tiling_key.h"

namespace NsGcd {

using namespace AscendC;
using half = AscendC::LocalTensor<float16>::ElementType;

static __aicore__ inline half GcdScalar(half a, half b)
{
    a = (a < (half)0.0f) ? -a : a;
    b = (b < (half)0.0f) ? -b : b;
    if (a == (half)0.0f) return b;
    if (b == (half)0.0f) return a;
    while (a != b) {
        if (a > b) {
            a = a - b;
        } else {
            b = b - a;
        }
    }
    return a;
}

static __aicore__ inline void DecomposeIndex(int64_t flat, const int64_t* strides,
                                              int64_t* indices, int64_t dimNum)
{
    int64_t rem = flat;
    for (int64_t d = 0; d < dimNum; d++) {
        if (strides[d] > 0) {
            indices[d] = rem / strides[d];
            rem = rem % strides[d];
        } else {
            indices[d] = 0;
        }
    }
}

static __aicore__ inline int64_t FlattenIndex(const int64_t* indices,
                                               const int64_t* strides, int64_t dimNum)
{
    int64_t flat = 0;
    for (int64_t d = 0; d < dimNum; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}

template <typename T, int BUFFER_MODE>
class Gcd {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Gcd(){};
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                 const GcdTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeBlock(int64_t blockStart, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueOut;

    GlobalTensor<T> selfGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;

    const GcdTilingData* tiling_;
    int64_t blockLength_;
    int64_t ubLength_;

    int64_t dimNum_;
    int64_t selfShape_[GCD_MAX_SHAPE_DIMS];
    int64_t otherShape_[GCD_MAX_SHAPE_DIMS];
    int64_t outShape_[GCD_MAX_SHAPE_DIMS];
    int64_t selfStrides_[GCD_MAX_SHAPE_DIMS];
    int64_t otherStrides_[GCD_MAX_SHAPE_DIMS];
    int64_t outStrides_[GCD_MAX_SHAPE_DIMS];
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                                  const GcdTilingData* tilingData)
{
    tiling_ = tilingData;
    int64_t remainder = tiling_->totalNum - tiling_->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainder > tiling_->blockFactor) ? tiling_->blockFactor : remainder;
    ubLength_ = tiling_->ubFactor;

    selfGM.SetGlobalBuffer((__gm__ T*)self + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    otherGM.SetGlobalBuffer((__gm__ T*)other + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outGM.SetGlobalBuffer((__gm__ T*)out + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueOther, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueOut, BUFFER_NUM, ubLength_ * sizeof(T));

    dimNum_ = tiling_->dimNum;
    for (int64_t d = 0; d < dimNum_; d++) {
        selfShape_[d] = tiling_->selfShape[d];
        otherShape_[d] = tiling_->otherShape[d];
        outShape_[d] = tiling_->outShape[d];
        selfStrides_[d] = tiling_->selfStrides[d];
        otherStrides_[d] = tiling_->otherStrides[d];
        outStrides_[d] = tiling_->outStrides[d];
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::ComputeBlock(int64_t blockStart, int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = inputQueueSelf.template AllocTensor<T>();
    AscendC::LocalTensor<T> otherLocal = inputQueueOther.template AllocTensor<T>();
    AscendC::LocalTensor<T> outLocal = outputQueueOut.template AllocTensor<T>();

    for (int64_t i = 0; i < currentNum; i++) {
        int64_t outFlat = blockStart + i;
        int64_t outIdx[GCD_MAX_SHAPE_DIMS];
        DecomposeIndex(outFlat, outStrides_, outIdx, dimNum_);

        int64_t selfIdx[GCD_MAX_SHAPE_DIMS];
        for (int64_t d = 0; d < dimNum_; d++) {
            selfIdx[d] = (selfShape_[d] == 1) ? 0 : outIdx[d];
        }
        int64_t selfFlat = FlattenIndex(selfIdx, selfStrides_, dimNum_);

        int64_t otherIdx[GCD_MAX_SHAPE_DIMS];
        for (int64_t d = 0; d < dimNum_; d++) {
            otherIdx[d] = (otherShape_[d] == 1) ? 0 : outIdx[d];
        }
        int64_t otherFlat = FlattenIndex(otherIdx, otherStrides_, dimNum_);

        half a = (half)selfGM(selfFlat);
        half b = (half)otherGM(otherFlat);
        half result = GcdScalar(a, b);
        outLocal.SetValue(i, (T)result);
    }

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM[blockStart], outLocal, copyParams);

    inputQueueSelf.FreeTensor(selfLocal);
    inputQueueOther.FreeTensor(otherLocal);
    outputQueueOut.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        int64_t blockStart = ubLength_ * i;
        ComputeBlock(blockStart, currentNum);
    }
}

} // namespace NsGcd
#endif // GCD_H
