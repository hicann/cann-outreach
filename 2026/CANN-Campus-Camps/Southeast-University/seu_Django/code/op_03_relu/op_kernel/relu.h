/*!
 * \file relu.h
 * \brief FP16/FP32 ReLU with double buffering and exact tail copies.
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
    __aicore__ inline Relu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y,
                                    const ReluTilingData* tilingData)
{
    ubLength_ = tilingData->ubFactor;
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum || ubLength_ <= 0) {
        blockLength_ = 0;
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if (count * sizeof(T) % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset], count);
    } else {
        constexpr uint32_t alignNum = 32 / sizeof(T);
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{
            true, 0, static_cast<uint8_t>(alignNum - count % alignNum), 0};
        DataCopyPad(xLocal, inputGMX[offset], copy, pad);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    auto xLocal = inputQueueX.DeQue<T>();
    auto yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if (count * sizeof(T) % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal, count);
    } else {
        // Copy only valid bytes; padding never reaches GM.
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copy);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    const int64_t loopCount = blockLength_ / ubLength_ +
        (blockLength_ % ubLength_ != 0);
    int64_t currentNum = blockLength_ < ubLength_ ? blockLength_ : ubLength_;
    CopyIn(0, currentNum);
    for (int64_t i = 0; i < loopCount; ++i) {
        // Prefetch into the second buffer before computing the current tile.
        int64_t nextNum = 0;
        if (i + 1 < loopCount) {
            const int64_t remaining = blockLength_ - (i + 1) * ubLength_;
            nextNum = remaining < ubLength_ ? remaining : ubLength_;
            CopyIn(i + 1, nextNum);
        }
        Compute(currentNum);
        CopyOut(i, currentNum);
        currentNum = nextNum;
    }
}
} // namespace NsRelu
#endif // RELU_H
