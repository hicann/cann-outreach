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
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    int64_t totalNum = tilingData->totalNum;
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t start = blockIdx * this->blockLength_;

    if (start >= totalNum) {
        this->blockLength_ = 0;
    } else {
        int64_t remain = totalNum - start;
        if (remain < this->blockLength_) {
            this->blockLength_ = remain;
        }
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        // 由于 tiling 保证了 blockLength_ 是 ubLength_ 的整数倍，此处 currentNum 恒等于 ubLength_
        int64_t currentNum = this->ubLength_;
        if (i == loopCount - 1 && this->blockLength_ % this->ubLength_ != 0) {
            currentNum = this->blockLength_ % this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 直接使用 currentNum 计算，此时 currentNum 一定是 ALIGN_SIZE 的倍数（由 tiling 保证）
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsRelu
#endif // RELU_H