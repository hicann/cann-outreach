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

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

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
    blockLength_ =
        tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;

    /*
     * 当前：
     *
     * blockLength_ = 2048
     * ubLength_    = 2048
     *
     * 每个 AIV 负责一个连续的 2048 元素区间。
     */
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
        GetBlockIdx() * blockLength_,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
        GetBlockIdx() * blockLength_,
        blockLength_);

    /*
     * Double Buffer。
     */
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
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress],
        currentNum);

    inputQueueX.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    AscendC::Relu(
        yLocal,
        xLocal,
        currentNum);

    outputQueueY.EnQue<T>(yLocal);

    inputQueueX.FreeTensor(xLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress],
        yLocal,
        currentNum);

    outputQueueY.FreeTensor(yLocal);
}


template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t progress = 0;

    while (progress < blockLength_) {

        int64_t currentNum =
            blockLength_ - progress;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(
            progress,
            currentNum);

        Compute(currentNum);

        CopyOut(
            progress,
            currentNum);

        progress += currentNum;
    }
}

} // namespace NsRelu

#endif // RELU_H