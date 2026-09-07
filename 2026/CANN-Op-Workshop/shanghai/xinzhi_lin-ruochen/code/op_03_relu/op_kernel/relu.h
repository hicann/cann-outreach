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
    ubLength_ = tilingData->ubFactor;
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (blockOffset >= tilingData->totalNum || ubLength_ <= 0) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - blockOffset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    constexpr uint32_t alignNum = 32 / sizeof(T);
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if (count % alignNum == 0) {
        DataCopy(xLocal, inputGMX[offset], count);
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(alignNum - count % alignNum), static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.template EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    // Qualify the API to distinguish it from the enclosing Relu class.
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    const int64_t offset = progress * ubLength_;
    const uint32_t count = static_cast<uint32_t>(currentNum);
    constexpr uint32_t alignNum = 32 / sizeof(T);
    if (count % alignNum == 0) {
        DataCopy(outputGMY[offset], yLocal, count);
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t tileNum = blockLength_ / ubLength_ + (blockLength_ % ubLength_ != 0);
    const int64_t firstNum = blockLength_ < ubLength_ ? blockLength_ : ubLength_;
    CopyIn(0, firstNum);
    for (int64_t progress = 0; progress < tileNum; ++progress) {
        const int64_t remaining = blockLength_ - progress * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        // Fill the second input buffer while the current tile is being processed.
        if (progress + 1 < tileNum) {
            const int64_t nextRemaining = remaining - ubLength_;
            const int64_t nextNum = nextRemaining < ubLength_ ? nextRemaining : ubLength_;
            CopyIn(progress + 1, nextNum);
        }
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
