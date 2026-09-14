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

    GM_ADDR xGmAddr_;
    GM_ADDR yGmAddr_;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    blockIdx_ = AscendC::GetBlockIdx();

    // 存储 GM 地址（注意：GM_ADDR 本身可以用作地址计算）
    xGmAddr_ = x;
    yGmAddr_ = y;

    // 初始化队列缓冲区（双缓冲，depth=2）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 计算当前 block 内 GM 起始地址
    // block 内偏移 = blockIdx_ * blockLength_ + progress
    int64_t offset = blockIdx_ * blockLength_ + progress;
    GlobalTensor<T> localGMX;
    localGMX.SetGlobalBuffer((__gm__ T*)xGmAddr_ + offset, currentNum);

    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, localGMX, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xBuf = inputQueueX.DeQue<T>();
    LocalTensor<T> yBuf = outputQueueY.AllocTensor<T>();

    // ReLU: y = max(0, x)
    // 使用 AscendC::Relu 向量指令
    AscendC::Relu(yBuf, xBuf, static_cast<int32_t>(currentNum));

    inputQueueX.FreeTensor(xBuf);
    outputQueueY.EnQue(yBuf);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // 计算当前 block 内 GM 起始地址
    int64_t offset = blockIdx_ * blockLength_ + progress;
    GlobalTensor<T> localGMY;
    localGMY.SetGlobalBuffer((__gm__ T*)yGmAddr_ + offset, currentNum);

    LocalTensor<T> yResult = outputQueueY.DeQue<T>();
    DataCopy(localGMY, yResult, currentNum);
    outputQueueY.FreeTensor(yResult);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t totalNum = blockLength_;
    int64_t loopCount = (totalNum + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t progress = i * ubLength_;
        int64_t currentNum = (i == loopCount - 1) ? (totalNum - progress) : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
