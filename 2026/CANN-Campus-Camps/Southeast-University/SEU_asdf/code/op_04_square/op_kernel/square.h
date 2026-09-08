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

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remain = tilingData->totalNum - blockOffset;

    blockLength_ = remain > 0 ? (remain < tilingData->blockFactor ? remain : tilingData->blockFactor) : 0;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + blockOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };
    DataCopyPadExtParams<T> padParams{
        false,
        0,
        0,
        static_cast<T>(0)
    };

    DataCopyPad(inputLocal, inputGMX[offset], copyParams, padParams);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    AscendC::Mul(outputLocal, inputLocal, inputLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(outputGMY[offset], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t offset = i * ubLength_;
        const int64_t remain = blockLength_ - offset;
        const int64_t currentNum = remain < ubLength_ ? remain : ubLength_;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H