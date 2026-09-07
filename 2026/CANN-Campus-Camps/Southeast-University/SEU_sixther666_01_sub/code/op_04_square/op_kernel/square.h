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
    __aicore__ inline void CopyInTail(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOutTail(int64_t progress, int64_t currentNum);
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
    const int64_t blockStart = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    blockLength_ = blockStart < tilingData->totalNum
                       ? (tilingData->totalNum - blockStart < tilingData->blockFactor
                              ? tilingData->totalNum - blockStart
                              : tilingData->blockFactor)
                       : 0;
    ubLength_ = tilingData->ubFactor;

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_x) + blockStart, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + blockStart, blockLength_);

    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyInTail(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{};
    DataCopyPad(inputLocal, inputGMX[progress * ubLength_], copyParams, padParams);
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
    DataCopy(outputGMY[progress * ubLength_], outputLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOutTail(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t fullTileCount = blockLength_ / ubLength_;
    for (int64_t progress = 0; progress < fullTileCount; ++progress) {
        CopyIn(progress, ubLength_);
        Compute(ubLength_);
        CopyOut(progress, ubLength_);
    }

    const int64_t tailNum = blockLength_ - fullTileCount * ubLength_;
    if (tailNum > 0) {
        if ((tailNum * static_cast<int64_t>(sizeof(T))) % DATA_BLOCK_BYTES == 0) {
            CopyIn(fullTileCount, tailNum);
            Compute(tailNum);
            CopyOut(fullTileCount, tailNum);
        } else {
            CopyInTail(fullTileCount, tailNum);
            Compute(tailNum);
            CopyOutTail(fullTileCount, tailNum);
        }
    }
}

} // namespace NsSquare
#endif // SQUARE_H
