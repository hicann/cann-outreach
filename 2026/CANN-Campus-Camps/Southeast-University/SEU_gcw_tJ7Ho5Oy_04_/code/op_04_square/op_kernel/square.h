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

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAligned(
        int64_t progress);

    __aicore__ inline void CopyInTail(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOutAligned(
        int64_t progress);

    __aicore__ inline void CopyOutTail(
        int64_t progress,
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM> inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    int64_t blockOffset =
        static_cast<int64_t>(
            GetBlockIdx()) *
        tilingData->blockFactor;

    int64_t remaining =
        static_cast<int64_t>(
            tilingData->totalNum) -
        blockOffset;

    blockLength_ =
        remaining < tilingData->blockFactor
            ? remaining
            : tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output + blockOffset,
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
__aicore__ inline void Square<T>::CopyInAligned(
    int64_t progress)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        inputLocal,
        inputGMX[progress * ubLength_],
        ubLength_);

    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyInTail(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPadExtParams<T> padParams{
        false,
        0,
        0,
        0
    };

    DataCopyPad(
        inputLocal,
        inputGMX[progress * ubLength_],
        copyParams,
        padParams);

    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        currentNum);

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOutAligned(
    int64_t progress)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        outputLocal,
        ubLength_);

    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOutTail(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(
        outputGMY[progress * ubLength_],
        outputLocal,
        copyParams);

    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    int64_t fullTileNum =
        blockLength_ / ubLength_;

    int64_t tailNum =
        blockLength_ -
        fullTileNum * ubLength_;

    for (int64_t i = 0;
         i < fullTileNum;
         i++) {
        CopyInAligned(i);
        Compute(ubLength_);
        CopyOutAligned(i);
    }

    if (tailNum > 0) {
        CopyInTail(
            fullTileNum,
            tailNum);

        Compute(tailNum);

        CopyOutTail(
            fullTileNum,
            tailNum);
    }
}

}

#endif