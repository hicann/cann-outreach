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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 每个核负责连续 blockFactor 个元素：从 GetBlockIdx()*blockFactor 处开始
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockLength_, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockIdx * blockLength_, blockLength_);

    // 为 VECIN/VECOUT 队列分配 UB 缓冲（每个队列 BUFFER_NUM 个，每个能放 ubFactor 个元素）
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<int32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU(x) = max(x, 0)，使用 vector 指令 Maxs
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    // 本核元素按 ubFactor 切成多 tile 处理，TQue 双缓冲(DoubleBuffer)由 BUFFER_NUM=2 提供
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        int64_t currentNum = blockLength_ - progress * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
