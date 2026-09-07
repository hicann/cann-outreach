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
    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t actualBlockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
    int64_t remain = totalNum_ - blockOffset;
    actualBlockLength_ = remain > blockLength_ ? blockLength_ : remain;
    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + blockOffset, actualBlockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + blockOffset, actualBlockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Mul(yLocal, xLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    int64_t loopCount = (actualBlockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t remain = actualBlockLength_ - i * ubLength_;
        int64_t currentNum = remain > ubLength_ ? ubLength_ : remain;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}
} // namespace NsSquare
#endif // SQUARE_H
