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

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

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
    // 当前核起始位置
    int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) *
        tilingData->blockFactor;

    // 当前核实际需要处理的数据量
    int64_t remainNum =
        tilingData->totalNum - blockOffset;

    if (remainNum > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    } else {
        blockLength_ = remainNum;
    }

    // 每次搬入 UB 的最大元素数量
    ubLength_ = tilingData->ubFactor;

    // 设置当前核对应的 GM 地址
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    // Double Buffer
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
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        inputLocal,
        inputGMX[progress * ubLength_],
        currentNum);

    inputQueueX.EnQue(inputLocal);
}

// ============================================================
// Compute
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // y = max(0, x)
    AscendC::Relu(
        outputLocal,
        inputLocal,
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}

// ============================================================
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        outputLocal,
        currentNum);

    outputQueueY.FreeTensor(outputLocal);
}

// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 当前核需要循环处理多少次
    int64_t loopCount =
        (blockLength_ + ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t remainNum =
            blockLength_ - i * ubLength_;

        int64_t currentNum =
            remainNum > ubLength_
                ? ubLength_
                : remainNum;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H