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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData);
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
    int64_t currentBlockLength_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    int64_t totalNum = tilingData->totalNum;
    int64_t blockIdx = GetBlockIdx();
    int64_t startOffset = blockIdx * blockLength_;

    if (startOffset >= totalNum) {
        currentBlockLength_ = 0;
    } else {
        int64_t remain = totalNum - startOffset;
        currentBlockLength_ = remain < blockLength_ ? remain : blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, currentBlockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, currentBlockLength_);

    // 缓冲区按 ubLength_ 分配，ubLength_ 已在 tiling 侧对齐到 32B
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 用 DataCopyPad 支持尾部非 32B 对齐
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0, 0, 0
    };
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Mul(yLocal, xLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0, 0, 0
    };
    DataCopyPad(outputGMY[progress], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (currentBlockLength_ <= 0) {
        return;
    }
    int64_t offset = 0;
    while (offset < currentBlockLength_) {
        int64_t currentNum = currentBlockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsSquare
#endif // SQUARE_H