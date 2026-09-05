// /*!
//  * \file relu.h
//  * \brief Relu 算子 kernel 类定义
//  */

// #ifndef RELU_H
// #define RELU_H

// #include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "relu_tiling_data.h"
// #include "relu_tiling_key.h"

// namespace NsRelu {

// using namespace AscendC;

// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class Relu {
// public:
//     __aicore__ inline Relu(){};

//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
//     __aicore__ inline void Process();

// private:
//     __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
//     __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
//     __aicore__ inline void Compute(int64_t currentNum);

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

//     GlobalTensor<T> inputGMX;
//     GlobalTensor<T> outputGMY;

//     int64_t blockLength_ = 0;
//     int64_t ubLength_ = 0;
// };

// // TODO: 实现具体的 kernel 逻辑
// template <typename T>
// __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
// {
//     // TODO: 实现 Init 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyIn 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Compute(int64_t currentNum)
// {
//     // TODO: 实现 Compute 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyOut 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Process()
// {
//     // TODO: 实现 Process 逻辑
// }

// } // namespace NsRelu
// #endif // RELU_H


/*!
 * \file relu.h
 * \brief Relu 算子 Kernel 模板实现
 */

/*!
 * \file relu.h
 * \brief Relu 算子 Kernel 实现
 */

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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
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

    int64_t totalLength_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalLength_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t blockOffset = GetBlockIdx() * blockLength_;
    int64_t gmLength = blockOffset < totalLength_ ? totalLength_ - blockOffset : 0;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, gmLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, gmLength);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (totalLength_ <= 0 || blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t blockOffset = GetBlockIdx() * blockLength_;
    if (blockOffset >= totalLength_) {
        return;
    }

    int64_t currentBlockLength = blockLength_;
    if (blockOffset + currentBlockLength > totalLength_) {
        currentBlockLength = totalLength_ - blockOffset;
    }

    int64_t loopCount = (currentBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = currentBlockLength - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
