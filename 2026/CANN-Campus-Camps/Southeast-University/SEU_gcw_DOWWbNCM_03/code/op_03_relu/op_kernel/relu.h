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

constexpr int32_t BUFFER_NUM = 1;

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
    int64_t currentLength_ = 0;
    int64_t loopCount_ = 0;     
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    int64_t startIndex = this->blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());

    int64_t remainder = static_cast<int64_t>(tilingData->totalNum) - startIndex;
    this->currentLength_ = (remainder > 0) ? ((remainder < this->blockLength_) ? remainder : this->blockLength_) : 0;

    this->loopCount_ = (this->ubLength_ > 0) ? ((this->currentLength_ + this->ubLength_ - 1) / this->ubLength_) : 0;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startIndex, this->currentLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startIndex, this->currentLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal); 
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (this->currentLength_ <= 0) {
        return;
    }
    int64_t loopCount = this->loopCount_;

    for (int64_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i, this->ubLength_);
        Compute(this->ubLength_);
        CopyOut(i, this->ubLength_);
    }

    int64_t lastOffset = (loopCount - 1) * this->ubLength_;
    int64_t lastNum = this->currentLength_ - lastOffset;
    CopyIn(loopCount - 1, lastNum);
    Compute(lastNum);
    CopyOut(loopCount - 1, lastNum);
}

} // namespace NsRelu
#endif // RELU_H
