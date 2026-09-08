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
constexpr int64_t DATA_BLOCK_BYTES = 32;

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

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (blockOffset >= tilingData->totalNum) {
        return;
    }

    blockLength_ = tilingData->totalNum - blockOffset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + blockOffset, blockLength_);

    // Two queues with two buffers each allow MTE2, Vector and MTE3 to overlap.
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    if ((copyBytes % DATA_BLOCK_BYTES) == 0) {
        DataCopy(inputLocal, inputGMX[progress], currentNum);
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    Mul(outputLocal, inputLocal, inputLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    if ((copyBytes % DATA_BLOCK_BYTES) == 0) {
        DataCopy(outputGMY[progress], outputLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPad(outputGMY[progress], outputLocal, copyParams);
    }
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
