/*!
 * \file square.h
 * \brief Square 算子 kernel 实现
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
    __aicore__ inline Square() {}

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);

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

    int64_t tailNum_ = 0;
};


// ================================================================
// Init
// ================================================================

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    totalNum_ =
        tilingData->totalNum;

    blockLength_ =
        tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;

    tailNum_ =
        tilingData->tailNum;

    const int64_t blockIdx =
        static_cast<int64_t>(GetBlockIdx());

    // ------------------------------------------------------------
    // 当前 Core 对应的 GM 起始位置
    // ------------------------------------------------------------

    const int64_t blockOffset =
        blockIdx * blockLength_;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(input_x) +
        blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(output) +
        blockOffset,
        blockLength_);

    // ------------------------------------------------------------
    // 双 Buffer
    //
    // Input:
    //     Buffer 0
    //     Buffer 1
    //
    // Output:
    //     Buffer 0
    //     Buffer 1
    //
    // 允许搬运和计算形成流水
    // ------------------------------------------------------------

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ================================================================
// CopyIn
// ================================================================

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    const int64_t offset =
        progress * ubLength_;

    const int64_t bytes =
        currentNum * sizeof(T);

    // ------------------------------------------------------------
    // 最常见情况：
    // currentNum * sizeof(T) 是 32B 对齐的
    //
    // 直接使用 DataCopy
    // ------------------------------------------------------------

    if ((bytes & 31) == 0) {

        DataCopy(
            inputLocal,
            inputGMX[offset],
            currentNum);

    } else {

        // --------------------------------------------------------
        // 只有最后一个非对齐 Tile 才会进入这里
        // --------------------------------------------------------

        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        copyParams.blockLen =
            static_cast<uint32_t>(bytes);

        copyParams.srcStride = 0;

        copyParams.dstStride = 0;

        copyParams.rsv = 0;

        const int64_t alignElementNum =
            32 / sizeof(T);

        const int64_t alignedNum =
            ((currentNum +
              alignElementNum -
              1) /
             alignElementNum) *
            alignElementNum;

        const int64_t rightPadding =
            alignedNum - currentNum;

        DataCopyPadExtParams<T> padParams;

        padParams.isPad = true;

        padParams.leftPadding = 0;

        padParams.rightPadding =
            static_cast<uint8_t>(
                rightPadding);

        padParams.paddingValue =
            static_cast<T>(0);

        DataCopyPad(
            inputLocal,
            inputGMX[offset],
            copyParams,
            padParams);
    }

    inputQueueX.EnQue(inputLocal);
}


// ================================================================
// Compute
// ================================================================

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // ------------------------------------------------------------
    // 使用 Counter Mask
    //
    // 这样即使最后一个 Tile 不满 32B，
    // 也可以只计算有效元素。
    // ------------------------------------------------------------

    SetMaskCount();

    SetVectorMask<T, MaskMode::COUNTER>(
        static_cast<int32_t>(
            currentNum));

    Mul<T, false>(
        outputLocal,
        inputLocal,
        inputLocal,
        MASK_PLACEHOLDER,
        1,
        {1, 1, 1, 8, 8, 8});

    SetMaskNorm();

    ResetMask();

    outputQueueY.EnQue(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}


// ================================================================
// CopyOut
// ================================================================

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    const int64_t offset =
        progress * ubLength_;

    const int64_t bytes =
        currentNum * sizeof(T);

    // ------------------------------------------------------------
    // 对齐情况
    // ------------------------------------------------------------

    if ((bytes & 31) == 0) {

        DataCopy(
            outputGMY[offset],
            outputLocal,
            currentNum);

    } else {

        // --------------------------------------------------------
        // 最后的非对齐输出
        // --------------------------------------------------------

        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        copyParams.blockLen =
            static_cast<uint32_t>(bytes);

        copyParams.srcStride = 0;

        copyParams.dstStride = 0;

        copyParams.rsv = 0;

        DataCopyPad(
            outputGMY[offset],
            outputLocal,
            copyParams);
    }

    outputQueueY.FreeTensor(outputLocal);
}


// ================================================================
// Process
// ================================================================

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    const int64_t blockIdx =
        static_cast<int64_t>(
            GetBlockIdx());

    const int64_t blockNumTotal =
        static_cast<int64_t>(
            GetBlockNum());

    // ------------------------------------------------------------
    // 当前 Core 实际处理的数据量
    // ------------------------------------------------------------

    int64_t blockNum;

    if (blockIdx ==
        blockNumTotal - 1) {

        // 最后一个 Core
        blockNum = tailNum_;

    } else {

        // 普通 Core
        blockNum = blockLength_;
    }

    if (blockNum <= 0) {
        return;
    }

    // ------------------------------------------------------------
    // 当前 Core 需要处理多少个 Tile
    // ------------------------------------------------------------

    const int64_t loopCount =
        (blockNum + ubLength_ - 1) /
        ubLength_;

    // ------------------------------------------------------------
    // 主循环
    // ------------------------------------------------------------

    for (int64_t progress = 0;
         progress < loopCount;
         ++progress) {

        const int64_t offset =
            progress * ubLength_;

        const int64_t remain =
            blockNum - offset;

        const int64_t currentNum =
            (remain > ubLength_)
                ? ubLength_
                : remain;

        if (currentNum <= 0) {
            break;
        }

        // GM -> UB
        CopyIn(
            progress,
            currentNum);

        // UB -> Vector
        Compute(
            currentNum);

        // UB -> GM
        CopyOut(
            progress,
            currentNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H