/*!
 * \file selu_grad.h
 * \brief SeluGrad 算子 kernel 类定义
 */

#ifndef SELUGRAD_H
#define SELUGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "selu_grad_tiling_data.h"
#include "selu_grad_tiling_key.h"
#include <type_traits>

namespace NsSeluGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SeluGrad {
public:
    __aicore__ inline SeluGrad(){};

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR outputs, GM_ADDR y, const SeluGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ReadValue(const LocalTensor<T>& tensor, int64_t index) const;
    __aicore__ inline T CastFromFloat(float value) const;
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradientsQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> outputsQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> calcBuffer;
    TBuf<QuePosition::VECCALC> maskBuffer;

    GlobalTensor<T> gradientsGM;
    GlobalTensor<T> outputsGM;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t blockOffset_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void SeluGrad<T>::Init(GM_ADDR gradients, GM_ADDR outputs, GM_ADDR y, const SeluGradTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockOffset_ = tilingData->blockFactor * GetBlockIdx();
    int64_t remain = totalNum_ - blockOffset_;
    blockLength_ = remain > tilingData->blockFactor ? tilingData->blockFactor : remain;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    gradientsGM.SetGlobalBuffer((__gm__ T*)gradients, totalNum_);
    outputsGM.SetGlobalBuffer((__gm__ T*)outputs, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > 0) {
        pipe.InitBuffer(gradientsQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputsQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        int64_t maskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
        pipe.InitBuffer(maskBuffer, static_cast<uint32_t>(maskBytes));
        if constexpr (std::is_same<T, float>::value) {
            pipe.InitBuffer(calcBuffer, ubLength_ * sizeof(float));
        } else {
            pipe.InitBuffer(calcBuffer, ubLength_ * sizeof(float) * 4);
        }
    }
}

template <typename T>
__aicore__ inline float SeluGrad<T>::ReadValue(const LocalTensor<T>& tensor, int64_t index) const
{
    T value = tensor.GetValue(index);
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T SeluGrad<T>::CastFromFloat(float value) const
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToBfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
__aicore__ inline void SeluGrad<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> gradLocal = gradientsQueue.template AllocTensor<T>();
    LocalTensor<T> outLocal = outputsQueue.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradLocal, gradientsGM[offset], copyParams, padParams);
    DataCopyPad(outLocal, outputsGM[offset], copyParams, padParams);
    gradientsQueue.EnQue(gradLocal);
    outputsQueue.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void SeluGrad<T>::Compute(int64_t currentNum)
{
    constexpr float SCALE = 1.0507009873554804934f;
    constexpr float SCALE_ALPHA = 1.75809934084737686f;
    LocalTensor<T> gradLocal = gradientsQueue.template DeQue<T>();
    LocalTensor<T> outInputLocal = outputsQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    if (currentNum <= 64) {
        // Preserve the low overhead path for tiny tensors; vector setup dominates there.
        for (int64_t i = 0; i < currentNum; ++i) {
            float grad = ReadValue(gradLocal, i);
            float output = ReadValue(outInputLocal, i);
            float result = output >= 0.0f ? grad * SCALE : grad * (output + SCALE_ALPHA);
            yLocal.SetValue(i, CastFromFloat(result));
        }
        outputQueue.EnQue(yLocal);
        gradientsQueue.FreeTensor(gradLocal);
        outputsQueue.FreeTensor(outInputLocal);
        return;
    }

    int64_t computeNum = ((currentNum + 63) / 64) * 64;
    LocalTensor<uint8_t> maskLocal = maskBuffer.Get<uint8_t>();
    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<float> negLocal = calcBuffer.Get<float>();
        Muls(yLocal, gradLocal, SCALE, static_cast<uint32_t>(computeNum));
        Adds(negLocal, outInputLocal, SCALE_ALPHA, static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Mul(negLocal, negLocal, gradLocal, static_cast<uint32_t>(computeNum));
        CompareScalar(maskLocal, outInputLocal, 0.0f, CMPMODE::GE, static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Select(yLocal, maskLocal, yLocal, negLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(computeNum));
    } else {
        LocalTensor<float> gradFloat = calcBuffer.GetWithOffset<float>(ubLength_, 0);
        LocalTensor<float> outFloat = calcBuffer.GetWithOffset<float>(ubLength_, ubLength_ * sizeof(float));
        LocalTensor<float> posFloat = calcBuffer.GetWithOffset<float>(ubLength_, ubLength_ * sizeof(float) * 2);
        LocalTensor<float> negFloat = calcBuffer.GetWithOffset<float>(ubLength_, ubLength_ * sizeof(float) * 3);
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(computeNum));
        Cast(outFloat, outInputLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Muls(posFloat, gradFloat, SCALE, static_cast<uint32_t>(computeNum));
        Adds(negFloat, outFloat, SCALE_ALPHA, static_cast<uint32_t>(computeNum));
        CompareScalar(maskLocal, outFloat, 0.0f, CMPMODE::GE, static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Mul(negFloat, negFloat, gradFloat, static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Select(posFloat, maskLocal, posFloat, negFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(computeNum));
        PipeBarrier<PIPE_V>();
        Cast(yLocal, posFloat, RoundMode::CAST_RINT, static_cast<uint32_t>(computeNum));
    }
    PipeBarrier<PIPE_V>();
    outputQueue.EnQue(yLocal);
    gradientsQueue.FreeTensor(gradLocal);
    outputsQueue.FreeTensor(outInputLocal);
}

template <typename T>
__aicore__ inline void SeluGrad<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[offset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SeluGrad<T>::Process()
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

} // namespace NsSeluGrad
#endif // SELUGRAD_H
