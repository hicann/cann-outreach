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
    __aicore__ inline Relu() {};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
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
    // 获取 Tiling 参数
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    if (totalNum_ <= 0) {
        totalNum_ = 0;
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    if (blockLength_ <= 0) {
        blockLength_ = 1;
    }

    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 当前 AI Core
    const int64_t blockIdx =
        static_cast<int64_t>(GetBlockIdx());

    // 当前 Core 对应的起始位置
    const int64_t blockOffset =
        blockIdx * blockLength_;

    // 当前 Core 实际起始地址
    __gm__ T* inputPtr =
        reinterpret_cast<__gm__ T*>(x) + blockOffset;

    __gm__ T* outputPtr =
        reinterpret_cast<__gm__ T*>(y) + blockOffset;

    // 初始化 GM Tensor
    inputGMX.SetGlobalBuffer(
        inputPtr,
        static_cast<uint32_t>(blockLength_));

    outputGMY.SetGlobalBuffer(
        outputPtr,
        static_cast<uint32_t>(blockLength_));

    // DoubleBuffer
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));
}


// ============================================================
// CopyIn
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    if (currentNum <= 0) {
        return;
    }

    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress * ubLength_],
        static_cast<uint32_t>(currentNum));

    inputQueueX.EnQue(xLocal);
}


// ============================================================
// Compute
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    if (currentNum <= 0) {
        return;
    }

    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // ReLU:
    //
    // y = max(0, x)
    //
    // Ascend C Maxs:
    // dst = max(src, scalar)
    Maxs(
        yLocal,
        xLocal,
        static_cast<T>(0),
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}


// ============================================================
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    if (currentNum <= 0) {
        return;
    }

    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        yLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.FreeTensor(yLocal);
}


// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (totalNum_ <= 0 ||
        blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }

    const int64_t blockIdx =
        static_cast<int64_t>(GetBlockIdx());

    // 当前 Core 的真实起始位置
    const int64_t blockOffset =
        blockIdx * blockLength_;

    // 已经超出数据范围
    if (blockOffset >= totalNum_) {
        return;
    }

    // 当前 Core 实际处理的数据量
    int64_t currentBlockNum =
        blockLength_;

    const int64_t remain =
        totalNum_ - blockOffset;

    if (remain < currentBlockNum) {
        currentBlockNum = remain;
    }

    if (currentBlockNum <= 0) {
        return;
    }

    // UB 循环次数
    const int64_t loopCount =
        (currentBlockNum + ubLength_ - 1) /
        ubLength_;

    // ----------------------------------------------------------
    // DoubleBuffer 流水
    //
    // 第一个 tile 先 CopyIn
    // 后续：
    // CopyIn(next)
    // Compute(current)
    // CopyOut(current)
    //
    // 最后再处理最后一个 tile
    // ----------------------------------------------------------

    int64_t currentNum =
        ubLength_;

    if (currentBlockNum < currentNum) {
        currentNum = currentBlockNum;
    }

    // 先加载第一个 tile
    CopyIn(0, currentNum);

    // 流水执行
    for (int64_t i = 1; i < loopCount; ++i) {

        // 当前需要加载的 tile 大小
        const int64_t nextOffset =
            i * ubLength_;

        int64_t nextNum =
            currentBlockNum - nextOffset;

        if (nextNum > ubLength_) {
            nextNum = ubLength_;
        }

        // CopyIn 下一块
        CopyIn(i, nextNum);

        // Compute 当前块
        Compute(currentNum);

        // CopyOut 当前块
        CopyOut(i - 1, currentNum);

        currentNum = nextNum;
    }

    // 处理最后一个 tile
    Compute(currentNum);

    CopyOut(
        loopCount - 1,
        currentNum);
}

} // namespace NsRelu

#endif // RELU_H