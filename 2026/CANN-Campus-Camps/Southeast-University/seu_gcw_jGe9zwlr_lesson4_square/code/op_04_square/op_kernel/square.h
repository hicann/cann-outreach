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

    __aicore__ inline Square() {};

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);

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

    // 整个 Tensor 的元素数量
    int64_t totalNum_ = 0;

    // 当前 Core 负责的元素数量
    int64_t blockLength_ = 0;

    // 每次 UB 处理的元素数量
    int64_t ubLength_ = 0;
};


// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;

    ubLength_ = tilingData->ubFactor;

    // 每个 Core 的理论最大处理长度
    int64_t blockFactor =
        tilingData->blockFactor;

    int64_t blockIdx =
        GetBlockIdx();

    // 当前 Core 在整个 Tensor 中的起始位置
    int64_t blockStart =
        blockIdx * blockFactor;

    // 当前 Core 实际需要处理多少元素
    if (blockStart >= totalNum_) {
        blockLength_ = 0;
    } else {
        int64_t remain =
            totalNum_ - blockStart;

        blockLength_ =
            (remain < blockFactor)
            ? remain
            : blockFactor;
    }

    // 设置当前 Core 对应的 GM 区域
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x + blockStart,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output + blockStart,
        blockLength_);

    // 初始化输入 / 输出 Queue
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
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    // 非 32 字节对齐场景使用 DataCopyPad
    DataCopyExtParams copyParams = {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPadExtParams<T> padParams = {
        false,
        0,
        0,
        0
    };

    DataCopyPad(
        inputLocal,
        inputGMX[progress],
        copyParams,
        padParams);

    inputQueueX.EnQue(inputLocal);
}


// ============================================================
// Compute
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // y = x * x
    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        currentNum);

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}


// ============================================================
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams = {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    // GM 地址可能不是 32B 对齐，所以同样使用 DataCopyPad
    DataCopyPad(
        outputGMY[progress],
        outputLocal,
        copyParams);

    outputQueueY.FreeTensor(outputLocal);
}


// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    // 当前 Core 负责的元素数量
    int64_t processed = 0;

    while (processed < blockLength_) {

        int64_t remain =
            blockLength_ - processed;

        int64_t currentNum =
            (remain > ubLength_)
            ? ubLength_
            : remain;

        CopyIn(
            processed,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            processed,
            currentNum);

        processed += currentNum;
    }
}

} // namespace NsSquare

#endif // SQUARE_H