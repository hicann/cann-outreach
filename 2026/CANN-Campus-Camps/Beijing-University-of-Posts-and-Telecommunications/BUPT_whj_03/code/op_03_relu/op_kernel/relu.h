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

__aicore__ constexpr TQueConfig MakeReluQueueConfig()
{
    TQueConfig config{};
    config.bufferNumber = BUFFER_NUM;
    return config;
}

static constexpr TQueConfig RELU_QUEUE_CONFIG = MakeReluQueueConfig();

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
    // 队列深度为 1，实际 buffer 数仍为 BUFFER_NUM，保持双缓冲。
    TQue<QuePosition::VECIN, 1, &RELU_QUEUE_CONFIG> inputQueueX;
    TQue<QuePosition::VECOUT, 1, &RELU_QUEUE_CONFIG> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blockFactor;
    const int64_t totalNum = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;
    blockLength_ = 0;
    if (offset >= totalNum) {
        return;
    }
    const int64_t remaining = totalNum - offset;
    blockLength_ = remaining < blockFactor ? remaining : blockFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 显式限定命名空间，避免与当前的 Relu 类名混淆。
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    const int64_t loopCount = blockLength_ / ubLength_ + (blockLength_ % ubLength_ != 0);
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t remaining = blockLength_ - i * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
