/*!
 * \file soft_shrink_grad.h
 * \brief SoftShrinkGrad 算子 kernel 类定义
 */

#ifndef SOFTSHRINKGRAD_H
#define SOFTSHRINKGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_shrink_grad_tiling_data.h"
#include "soft_shrink_grad_tiling_key.h"
#include <type_traits>

namespace NsSoftShrinkGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SoftShrinkGrad {
public:
    __aicore__ inline SoftShrinkGrad(){};

    __aicore__ inline void Init(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y, const SoftShrinkGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ReadValue(const LocalTensor<T>& tensor, int64_t index) const;
    __aicore__ inline T CastFromFloat(float value) const;
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputGradQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> calcBuffer;
    TBuf<QuePosition::VECCALC> maskBuffer;

    GlobalTensor<T> inputGradGM;
    GlobalTensor<T> inputXGM;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t blockOffset_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Init(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y, const SoftShrinkGradTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    lambd_ = tilingData->lambd;
    blockOffset_ = tilingData->blockFactor * GetBlockIdx();
    int64_t remain = totalNum_ - blockOffset_;
    blockLength_ = remain > tilingData->blockFactor ? tilingData->blockFactor : remain;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    inputGradGM.SetGlobalBuffer((__gm__ T*)input_grad, totalNum_);
    inputXGM.SetGlobalBuffer((__gm__ T*)input_x, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output_y, totalNum_);
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > 0) {
        pipe.InitBuffer(inputGradQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(inputXQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        int64_t maskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
        pipe.InitBuffer(maskBuffer, static_cast<uint32_t>(maskBytes));
        if constexpr (std::is_same<T, half>::value) {
            pipe.InitBuffer(calcBuffer, ubLength_ * sizeof(float));
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            pipe.InitBuffer(calcBuffer, ubLength_ * sizeof(float) * 3);
        }
    }
}

template <typename T>
__aicore__ inline float SoftShrinkGrad<T>::ReadValue(const LocalTensor<T>& tensor, int64_t index) const
{
    T value = tensor.GetValue(index);
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T SoftShrinkGrad<T>::CastFromFloat(float value) const
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToBfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputGradQueue.template AllocTensor<T>();
    LocalTensor<T> xLocal = inputXQueue.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradLocal, inputGradGM[offset], copyParams, padParams);
    DataCopyPad(xLocal, inputXGM[offset], copyParams, padParams);
    inputGradQueue.EnQue(gradLocal);
    inputXQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputGradQueue.template DeQue<T>();
    LocalTensor<T> xLocal = inputXQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    if (currentNum <= 64) {
        // Small tensors are launch-latency dominated; the scalar path avoids extra compare/select setup.
        for (int64_t i = 0; i < currentNum; ++i) {
            float x = ReadValue(xLocal, i);
            float grad = ReadValue(gradLocal, i);
            float result = (x > lambd_ || x < -lambd_) ? grad : 0.0f;
            yLocal.SetValue(i, CastFromFloat(result));
        }
        outputQueue.EnQue(yLocal);
        inputGradQueue.FreeTensor(gradLocal);
        inputXQueue.FreeTensor(xLocal);
        return;
    }

    // SoftShrinkGrad keeps grad where x is outside [-lambd, lambd].
    // Compare in fp32 for low-bit dtypes so threshold handling matches the scalar reference.
    int64_t compareNum = ((currentNum + 63) / 64) * 64;
    LocalTensor<uint8_t> maskLocal = maskBuffer.Get<uint8_t>();
    if constexpr (std::is_same<T, float>::value) {
        CompareScalar(maskLocal, xLocal, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yLocal, maskLocal, gradLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xLocal, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yLocal, maskLocal, gradLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
    } else if constexpr (std::is_same<T, half>::value) {
        LocalTensor<float> xFloat = calcBuffer.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yLocal, maskLocal, gradLocal, static_cast<half>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yLocal, maskLocal, gradLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
    } else {
        LocalTensor<float> xFloat = calcBuffer.GetWithOffset<float>(ubLength_, 0);
        LocalTensor<float> gradFloat = calcBuffer.GetWithOffset<float>(ubLength_, ubLength_ * sizeof(float));
        LocalTensor<float> yFloat = calcBuffer.GetWithOffset<float>(ubLength_, ubLength_ * sizeof(float) * 2);
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, lambd_, CMPMODE::GT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yFloat, maskLocal, gradFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, -lambd_, CMPMODE::LT, static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Select(yFloat, maskLocal, gradFloat, yFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(compareNum));
        PipeBarrier<PIPE_V>();
        Cast(yLocal, yFloat, RoundMode::CAST_RINT, static_cast<uint32_t>(compareNum));
    }
    PipeBarrier<PIPE_V>();
    outputQueue.EnQue(yLocal);
    inputGradQueue.FreeTensor(gradLocal);
    inputXQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[offset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SoftShrinkGrad<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        int64_t offset = blockOffset_ + i * ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsSoftShrinkGrad
#endif // SOFTSHRINKGRAD_H
