/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
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
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    int64_t startOffset = blockIdx * tilingData->blockFactor;

    if (startOffset >= tilingData->totalNum) {
        blockLength_ = 0;
        ubLength_ = tilingData->ubFactor;
        return;
    }

    blockLength_ = tilingData->totalNum - startOffset;
    blockLength_ = blockLength_ > tilingData->blockFactor ? tilingData->blockFactor : blockLength_;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + startOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    constexpr int64_t ALIGN_NUM = 32 / sizeof(T);
    int64_t computeNum = (currentNum + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

    AscendC::Mul(yLocal, xLocal, xLocal, static_cast<uint32_t>(computeNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(outputGMY[progress], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t progress = i * ubLength_;
        int64_t currentNum = blockLength_ - progress;
        currentNum = currentNum > ubLength_ ? ubLength_ : currentNum;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
