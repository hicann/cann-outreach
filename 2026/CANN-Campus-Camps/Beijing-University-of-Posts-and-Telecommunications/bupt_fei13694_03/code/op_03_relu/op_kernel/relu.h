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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t blockIdx =
        static_cast<int64_t>(AscendC::GetBlockIdx());

    const int64_t blockOffset =
        blockIdx * tilingData->blockFactor;

    /*
     * 通常 blockLength_ 等于 blockFactor。
     * 最后一个核可能只处理剩余元素。
     */
    const int64_t remaining =
        tilingData->totalNum - blockOffset;

    blockLength_ =
        remaining < tilingData->blockFactor
            ? remaining
            : tilingData->blockFactor;

    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + blockOffset,
        blockLength_);

    /*
     * 为输入和输出队列分别分配两个 Buffer。
     *
     * 总 UB 使用量：
     *   2 个队列 × 2 个 Buffer × ubLength_ × sizeof(T)
     */
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
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    const int64_t offset =
        progress * ubLength_;

    DataCopy(
        xLocal,
        inputGMX[offset],
        static_cast<uint32_t>(currentNum));

    inputQueueX.EnQue<T>(xLocal);
}

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
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    const int64_t offset =
        progress * ubLength_;

    DataCopy(
        outputGMY[offset],
        yLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t progress = 0;
         progress < loopCount;
         ++progress) {
        const int64_t processed =
            progress * ubLength_;

        const int64_t remaining =
            blockLength_ - processed;

        const int64_t currentNum =
            remaining < ubLength_
                ? remaining
                : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
