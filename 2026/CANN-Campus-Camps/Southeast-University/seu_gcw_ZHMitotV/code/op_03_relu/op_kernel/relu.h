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
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

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
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();

    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t offset =
        blockIdx * tilingData->blockFactor;

    if (offset >= tilingData->totalNum) {
        blockLength_ = 0;
        return;
    }

    int64_t remain =
        tilingData->totalNum - offset;

    if (remain < blockLength_) {
        blockLength_ = remain;
    }

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + offset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + offset,
        blockLength_);

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        inputLocal,
        inputGMX[progress * ubLength_],
        currentNum);

    inputQueueX.EnQue(inputLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    AscendC::Relu(
        outputLocal,
        inputLocal,
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        outputLocal,
        currentNum);

    outputQueueY.FreeTensor(outputLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t tileNum =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < tileNum; ++i) {

        int64_t offset =
            i * ubLength_;

        int64_t currentNum =
            blockLength_ - offset;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(i, currentNum);

        Compute(currentNum);

        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H