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

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM
    > inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM
    > outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};


// ========================================
// Init
// ========================================
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    // 当前核在整个Tensor中的起始位置
    const int64_t blockOffset =
        tilingData->blockFactor *
        static_cast<int64_t>(
            AscendC::GetBlockIdx());

    // 剩余元素数量
    const int64_t remainderLength =
        tilingData->totalNum -
        blockOffset;

    // 最后一个核可能不足blockFactor
    blockLength_ =
        (remainderLength >
         tilingData->blockFactor)
            ? tilingData->blockFactor
            : remainderLength;

    ubLength_ =
        tilingData->ubFactor;

    // 设置GM地址
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
            blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
            blockOffset,
        blockLength_);

    // DoubleBuffer
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ========================================
// CopyIn
// ========================================
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX
            .AllocTensor<T>();

    DataCopyParams copyParams;

    copyParams.blockCount = 1;

    copyParams.blockLen =
        currentNum *
        sizeof(T);

    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(
        xLocal,
        inputGMX[
            progress *
            ubLength_
        ],
        copyParams,
        {false, 0, 0, 0});

    inputQueueX.EnQue(
        xLocal);
}


// ========================================
// Compute
// ========================================
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX
            .DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY
            .AllocTensor<T>();

    // ==================================
    // ReLU核心运算
    // y = max(0, x)
    // ==================================
    AscendC::Relu(
        yLocal,
        xLocal,
        currentNum);

    outputQueueY.EnQue<T>(
        yLocal);

    inputQueueX.FreeTensor(
        xLocal);
}


// ========================================
// CopyOut
// ========================================
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY
            .DeQue<T>();

    DataCopyParams copyParams;

    copyParams.blockCount = 1;

    copyParams.blockLen =
        currentNum *
        sizeof(T);

    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(
        outputGMY[
            progress *
            ubLength_
        ],
        yLocal,
        copyParams);

    outputQueueY.FreeTensor(
        yLocal);
}


// ========================================
// Process
// ========================================
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }

    // 向上取整，兼容最后一块不足ubLength
    const int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0;
         i < loopCount;
         ++i)
    {
        // 最后一块单独计算实际元素数
        const int64_t currentNum =
            (i == loopCount - 1)
                ? (blockLength_ -
                   i * ubLength_)
                : ubLength_;

        CopyIn(
            i,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            i,
            currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
