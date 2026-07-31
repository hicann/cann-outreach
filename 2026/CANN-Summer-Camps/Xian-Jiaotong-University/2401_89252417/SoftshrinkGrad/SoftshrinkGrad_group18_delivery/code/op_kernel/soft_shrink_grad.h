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
 * \file soft_shrink_grad.h
 * \brief SoftShrinkGrad Ascend C kernel implementation
 */

#ifndef SOFTSHRINKGRAD_H
#define SOFTSHRINKGRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_shrink_grad_tiling_data.h"
#include "soft_shrink_grad_tiling_key.h"

namespace NsSoftShrinkGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t SMALL_TENSOR_THRESHOLD = 64;

template <typename T>
class SoftShrinkGrad {
public:
    __aicore__ inline SoftShrinkGrad() {};
    __aicore__ inline void Init(GM_ADDR inputGrad, GM_ADDR inputX, GM_ADDR outputY,
                                const SoftShrinkGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ReadAsFloat(const LocalTensor<T>& tensor, int64_t index) const;
    __aicore__ inline T CastFromFloat(float value) const;
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> computeBuffer;
    TBuf<QuePosition::VECCALC> maskBuffer;

    GlobalTensor<T> gradGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> outputGM;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t blockOffset_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Init(GM_ADDR inputGrad, GM_ADDR inputX, GM_ADDR outputY,
                                                const SoftShrinkGradTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    blockOffset_ = tilingData->blockFactor * GetBlockIdx();
    int64_t remaining = totalNum_ - blockOffset_;
    blockLength_ = remaining > tilingData->blockFactor ? tilingData->blockFactor : remaining;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    gradGM.SetGlobalBuffer((__gm__ T*)inputGrad, totalNum_);
    xGM.SetGlobalBuffer((__gm__ T*)inputX, totalNum_);
    outputGM.SetGlobalBuffer((__gm__ T*)outputY, totalNum_);

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    pipe.InitBuffer(gradQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    int64_t maskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
    pipe.InitBuffer(maskBuffer, static_cast<uint32_t>(maskBytes));

    if constexpr (std::is_same<T, half>::value) {
        pipe.InitBuffer(computeBuffer, ubLength_ * sizeof(float));
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        pipe.InitBuffer(computeBuffer, ubLength_ * sizeof(float) * 3);
    }
}

template <typename T>
__aicore__ inline float SoftShrinkGrad<T>::ReadAsFloat(const LocalTensor<T>& tensor, int64_t index) const
{
    T value = tensor.GetValue(index);
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToFloat(value);
    }
    return static_cast<float>(value);
}

template <typename T>
__aicore__ inline T SoftShrinkGrad<T>::CastFromFloat(float value) const
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToBfloat16(value);
    }
    return static_cast<T>(value);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> gradLocal = gradQueue.template AllocTensor<T>();
    LocalTensor<T> xLocal = xQueue.template AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradLocal, gradGM[offset], copyParams, padParams);
    DataCopyPad(xLocal, xGM[offset], copyParams, padParams);

    gradQueue.EnQue(gradLocal);
    xQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> gradLocal = gradQueue.template DeQue<T>();
    LocalTensor<T> xLocal = xQueue.template DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    if (currentNum <= SMALL_TENSOR_THRESHOLD) {
        for (int64_t i = 0; i < currentNum; ++i) {
            float x = ReadAsFloat(xLocal, i);
            float grad = ReadAsFloat(gradLocal, i);
            float result = (x > lambd_ || x < -lambd_) ? grad : 0.0f;
            outputLocal.SetValue(i, CastFromFloat(result));
        }
        outputQueue.EnQue(outputLocal);
        gradQueue.FreeTensor(gradLocal);
        xQueue.FreeTensor(xLocal);
        return;
    }

    int64_t compareNum = ((currentNum + 63) / 64) * 64;
    LocalTensor<uint8_t> maskLocal = maskBuffer.Get<uint8_t>();

    if constexpr (std::is_same<T, float>::value) {
        CompareScalar(maskLocal, xLocal, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputLocal, maskLocal, gradLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xLocal, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputLocal, maskLocal, gradLocal, outputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
    } else if constexpr (std::is_same<T, half>::value) {
        LocalTensor<float> xFloat = computeBuffer.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputLocal, maskLocal, gradLocal, static_cast<half>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputLocal, maskLocal, gradLocal, outputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
    } else {
        LocalTensor<float> xFloat = computeBuffer.GetWithOffset<float>(ubLength_, 0);
        LocalTensor<float> gradFloat =
            computeBuffer.GetWithOffset<float>(ubLength_, ubLength_ * static_cast<int64_t>(sizeof(float)));
        LocalTensor<float> outputFloat =
            computeBuffer.GetWithOffset<float>(ubLength_, ubLength_ * static_cast<int64_t>(sizeof(float)) * 2);

        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputFloat, maskLocal, gradFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(outputFloat, maskLocal, gradFloat, outputFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, static_cast<uint32_t>(compareNum));
    }

    PipeBarrier<PIPE_V>();
    outputQueue.EnQue(outputLocal);
    gradQueue.FreeTensor(gradLocal);
    xQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGM[offset], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = i == loopCount - 1 ? blockLength_ - i * ubLength_ : ubLength_;
        int64_t offset = blockOffset_ + i * ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsSoftShrinkGrad

#endif // SOFTSHRINKGRAD_H
