/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    uint32_t blockIdx = GetBlockIdx();
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t ubFactor = tilingData->ubFactor;

    int64_t start = (int64_t)blockIdx * blockFactor;
    int64_t end = (start + blockFactor > totalNum) ? totalNum : start + blockFactor;
    blockLength_ = (end > start) ? (end - start) : 0;

    ubLength_ = (ubFactor < blockLength_) ? ubFactor : blockLength_;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + start);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start);

    // 修正：TQue 没有 InitBuffer；pipe.InitBuffer 三参数：队列、buffer数、每块字节数
    pipe.InitBuffer(inputQueueX, (uint8_t)BUFFER_NUM, (uint32_t)(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, (uint8_t)BUFFER_NUM, (uint32_t)(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 修正：队列返回 LocalTensor<T>
    LocalTensor<T> inputX = inputQueueX.AllocTensor<T>();
    DataCopy(inputX, inputGMX[progress], currentNum);
    inputQueueX.EnQue(inputX);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputX = inputQueueX.DeQue<T>();
    LocalTensor<T> outputY = outputQueueY.AllocTensor<T>();
    AscendC::Relu(outputY, inputX, currentNum);
    inputQueueX.FreeTensor(inputX);
    outputQueueY.EnQue(outputY);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputY = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], outputY, currentNum);
    outputQueueY.FreeTensor(outputY);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    int64_t progress = 0;
    int64_t currentNum = (ubLength_ < blockLength_) ? ubLength_ : blockLength_;

    // Prologue
    CopyIn(progress, currentNum);

    // MainBody
    for (int64_t i = 1; i < loopCount; i++) {
        int64_t nextProgress = progress + currentNum;
        int64_t nextNum = (ubLength_ < blockLength_ - nextProgress)
                               ? ubLength_
                               : (blockLength_ - nextProgress);
        CopyIn(nextProgress, nextNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        progress = nextProgress;
        currentNum = nextNum;
    }

    // Epilogue
    Compute(currentNum);
    CopyOut(progress, currentNum);
}

} // namespace NsRelu
#endif // RELU_H
