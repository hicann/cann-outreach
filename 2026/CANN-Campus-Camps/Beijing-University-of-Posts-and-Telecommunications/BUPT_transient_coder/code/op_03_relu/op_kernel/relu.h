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

// 具体的 kernel 逻辑实现
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    this->ubLength_ = tilingData->ubFactor;

    int64_t coreOffset = blockIdx * tilingData->blockFactor;
    if (coreOffset >= tilingData->totalNum) {
        this->blockLength_ = 0;
        return;
    }

    int64_t curBlockLength = tilingData->blockFactor;
    if (coreOffset + curBlockLength > tilingData->totalNum) {
        curBlockLength = tilingData->totalNum - coreOffset;
    }
    this->blockLength_ = curBlockLength;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + coreOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + coreOffset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    constexpr uint32_t ALIGN_ELEMS = 32 / sizeof(T);
    uint32_t alignNum = (currentNum + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], alignNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    constexpr uint32_t ALIGN_ELEMS = 32 / sizeof(T);
    uint32_t alignNum = (currentNum + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
    AscendC::Relu(yLocal, xLocal, alignNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    constexpr uint32_t ALIGN_ELEMS = 32 / sizeof(T);
    uint32_t alignNum = (currentNum + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, alignNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (this->blockLength_ <= 0 || this->ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->ubLength_;
        if ((i + 1) * this->ubLength_ > this->blockLength_) {
            currentNum = this->blockLength_ - i * this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H

