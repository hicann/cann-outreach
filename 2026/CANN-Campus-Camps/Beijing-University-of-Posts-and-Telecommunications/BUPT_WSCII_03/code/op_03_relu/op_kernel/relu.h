#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

// 小 Tensor 单 Tile，不使用 DoubleBuffer
constexpr int32_t BUFFER_NUM = 1;

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

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
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
    int64_t blockOffset =
        tilingData->blockFactor *
        AscendC::GetBlockIdx();

    int64_t remain =
        tilingData->totalNum -
        blockOffset;

    blockLength_ =
        remain > tilingData->blockFactor
            ? tilingData->blockFactor
            : remain;

    ubLength_ =
        tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
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


// ============================================================
// CopyIn
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress * ubLength_],
        static_cast<uint32_t>(currentNum));

    inputQueueX.EnQue(xLocal);
}


// ============================================================
// Compute
// ============================================================

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
        static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}


// ============================================================
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        yLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.FreeTensor(yLocal);
}


// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    CopyIn(
        0,
        blockLength_);

    Compute(
        blockLength_);

    CopyOut(
        0,
        blockLength_);
}

} // namespace NsRelu

#endif // RELU_H