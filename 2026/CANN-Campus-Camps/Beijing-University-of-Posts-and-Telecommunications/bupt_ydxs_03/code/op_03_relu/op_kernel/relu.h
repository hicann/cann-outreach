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

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t blockStart_ = 0;
};

// ============================================================
// Init
// ============================================================
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x));

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y));

    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    blockStart_ = GetBlockIdx() * blockLength_;

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

// ============================================================
// CopyIn
// ============================================================
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        inputLocal,
        inputGMX[blockStart_ + progress * ubLength_],
        currentNum);

    inputQueueX.EnQue(inputLocal);
}

// ============================================================
// Compute
// ============================================================
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // ReLU:
    // y = max(x, 0)
    Maxs(
        outputLocal,
        inputLocal,
        static_cast<T>(0),
        currentNum);

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}

// ============================================================
// CopyOut
// ============================================================
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[blockStart_ + progress * ubLength_],
        outputLocal,
        currentNum);

    outputQueueY.FreeTensor(outputLocal);
}

// ============================================================
// Process
// ============================================================
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t totalLength = blockLength_;

    if (totalLength <= 0) {
        return;
    }

    int64_t loopCount =
        (totalLength + ubLength_ - 1) / ubLength_;

    for (int64_t progress = 0;
         progress < loopCount;
         ++progress) {

        int64_t processed =
            progress * ubLength_;

        int64_t currentNum = ubLength_;

        if (processed + currentNum > totalLength) {
            currentNum = totalLength - processed;
        }

        if (currentNum <= 0) {
            break;
        }

        CopyIn(
            progress,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            progress,
            currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H