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

// DoubleBuffer
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Square {
public:
    __aicore__ inline Square()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData *tilingData);

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

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM> inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前 Core 实际需要处理的数据数量
    int64_t blockLength_ = 0;

    // 每次 UB 最多处理的数据数量
    int64_t ubLength_ = 0;
};


// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData *tilingData)
{
    // 输入总元素数量
    int64_t totalNum =
        tilingData->totalNum;

    // Host 为每核规划的最大处理数量
    int64_t blockFactor =
        tilingData->blockFactor;

    // 每次 UB 处理的最大元素数
    ubLength_ =
        tilingData->ubFactor;

    // 当前核编号
    int64_t blockIdx =
        static_cast<int64_t>(
            GetBlockIdx());

    // 当前核对应的起始元素
    int64_t blockOffset =
        blockIdx *
        blockFactor;

    // 当前核剩余数据量
    int64_t remain =
        totalNum -
        blockOffset;

    // ========================================================
    // 最后一个 Core 可能不足 blockFactor
    //
    // blockLength_ = min(blockFactor, remain)
    // ========================================================
    if (remain > blockFactor) {
        blockLength_ =
            blockFactor;
    } else if (remain > 0) {
        blockLength_ =
            remain;
    } else {
        blockLength_ = 0;
    }

    // ========================================================
    // 设置当前 Core 的 GM 范围
    // ========================================================
    inputGMX.SetGlobalBuffer(
        (__gm__ T *)input_x +
            blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T *)output +
            blockOffset,
        blockLength_);

    // ========================================================
    // 初始化 DoubleBuffer
    //
    // ubLength_ 已由 Host 保证按 32B 对齐容量计算。
    // ========================================================
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
//
// GM -> UB
//
// 使用 DataCopyPad，而不是 DataCopy。
// 这是支持非 32Byte 对齐 case 的关键。
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    int64_t offset =
        progress *
        ubLength_;

    /*
     * DataCopyPad 的 blockLen 单位为 Byte。
     *
     * 例如：
     *
     * float32 currentNum = 7
     *
     * blockLen = 7 * 4
     *          = 28 Byte
     *
     * 即使不是 32 Byte 整数倍也可以正确搬运。
     */
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    /*
     * 不需要人工指定 padding 内容。
     *
     * Compute 只计算 currentNum 个有效元素，
     * 所以补齐区域的值不会参与计算。
     */
    DataCopyPadExtParams<T> padParams{
        false,
        0,
        0,
        static_cast<T>(0)
    };

    DataCopyPad(
        inputLocal,
        inputGMX[offset],
        copyParams,
        padParams);

    inputQueueX.EnQue(
        inputLocal);
}


// ============================================================
// Compute
//
// y = x * x
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // ========================================================
    // Square 核心：
    //
    // output = input * input
    // ========================================================
    AscendC::Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        static_cast<uint32_t>(
            currentNum));

    outputQueueY.EnQue(
        outputLocal);

    inputQueueX.FreeTensor(
        inputLocal);
}


// ============================================================
// CopyOut
//
// UB -> GM
//
// 同样使用 DataCopyPad，确保尾块非对齐输出正确。
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    int64_t offset =
        progress *
        ubLength_;

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(
        outputGMY[offset],
        outputLocal,
        copyParams);

    outputQueueY.FreeTensor(
        outputLocal);
}


// ============================================================
// Process
//
// 支持任意 blockLength。
// 最后一次循环自动处理尾块。
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    // ceil(blockLength / ubLength)
    int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        // 当前循环起始位置
        int64_t processed =
            i * ubLength_;

        // 尚未处理的数据
        int64_t remain =
            blockLength_ -
            processed;

        // ====================================================
        // 最后一块：
        //
        // currentNum = min(ubLength_, remain)
        // ====================================================
        int64_t currentNum =
            remain > ubLength_
                ? ubLength_
                : remain;

        CopyIn(
            i,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            i,
            currentNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H