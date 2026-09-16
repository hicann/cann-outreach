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

    int64_t offset = static_cast<int64_t>(blockIdx) * blockFactor;
    blockLength_ = blockFactor;
    if (offset + blockLength_ > totalNum) {
        blockLength_ = totalNum - offset;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    if (blockLength_ <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    int64_t ubBytes = ubLength_ * static_cast<int64_t>(sizeof(T));
    ubBytes = (ubBytes + 32 - 1) / 32 * 32; // 保证 32 字节对齐

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    // ReLU: y = max(x, 0)
    Maxs(outputLocal, inputLocal, (T)0, currentNum);

    inputQueueX.FreeTensor(inputLocal);
    outputQueueY.EnQue(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], outputLocal, currentNum);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t progress = 0;
    while (progress < blockLength_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);

        progress += currentNum;
    }
}

} // namespace NsRelu
#endif // RELU_H