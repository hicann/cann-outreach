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


// ============================================================
// Double Buffer
// ============================================================

constexpr int32_t BUFFER_NUM = 2;


// ============================================================
// Square Kernel
// ============================================================

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

    // Pipeline
    TPipe pipe;


    // 输入 Queue
    TQue<
        QuePosition::VECIN,
        BUFFER_NUM> inputQueueX;


    // 输出 Queue
    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM> outputQueueY;


    // Global Memory
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;


    // 当前 Core 真正负责多少元素
    int64_t blockLength_ = 0;


    // UB Tile 的分配长度
    // 该值在 Host 侧保证 32B 对齐
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
    // --------------------------------------------------------
    // 当前 Core 编号
    // --------------------------------------------------------

    int64_t blockIdx =
        static_cast<int64_t>(
            GetBlockIdx());


    // --------------------------------------------------------
    // 当前 Core 对应的 GM 起点
    // --------------------------------------------------------

    int64_t blockOffset =
        blockIdx *
        tilingData->blockFactor;


    // --------------------------------------------------------
    // 剩余元素数量
    // --------------------------------------------------------

    int64_t remainLength =
        tilingData->totalNum -
        blockOffset;


    // --------------------------------------------------------
    // 当前 Core 正常最多处理 blockFactor
    // --------------------------------------------------------

    blockLength_ =
        tilingData->blockFactor;


    // 最后一核可能不完整
    if (remainLength < blockLength_) {

        blockLength_ =
            remainLength;
    }


    // --------------------------------------------------------
    // UB 长度。
    //
    // 注意：
    //
    // 这里不能写：
    //
    // if (ubLength_ > blockLength_)
    //     ubLength_ = blockLength_;
    //
    // 因为 blockLength_ 可能不满足 32B 对齐。
    //
    // ubLength_ 必须继续保持 Host 侧算出的
    // 32B 对齐长度。
    // --------------------------------------------------------

    ubLength_ =
        tilingData->ubFactor;


    // --------------------------------------------------------
    // 当前 Core 输入 GM
    // --------------------------------------------------------

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x +
            blockOffset,
        blockLength_);


    // --------------------------------------------------------
    // 当前 Core 输出 GM
    // --------------------------------------------------------

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output +
            blockOffset,
        blockLength_);


    // --------------------------------------------------------
    // 输入 Double Buffer
    // --------------------------------------------------------

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));


    // --------------------------------------------------------
    // 输出 Double Buffer
    // --------------------------------------------------------

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ============================================================
// CopyIn
//
// GM -> Local
//
// 使用 DataCopyPad 支持非 32B 对齐数据。
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 申请 LocalTensor
    // --------------------------------------------------------

    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();


    // --------------------------------------------------------
    // 当前 Tile 在本 Core 内的位置
    // --------------------------------------------------------

    int64_t offset =
        progress *
        ubLength_;


    // --------------------------------------------------------
    // DataCopyPad 的 blockLen 单位是 Byte。
    //
    // 例如：
    //
    // float32 × 13
    //
    // blockLen =
    // 13 × 4 =
    // 52 Bytes
    //
    // 即使 52 不是 32 的整数倍，
    // DataCopyPad 也可以正确搬运。
    // --------------------------------------------------------

    DataCopyExtParams copyParams {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };


    // --------------------------------------------------------
    // 不人为增加业务数据，
    // 非对齐部分交给 DataCopyPad 处理。
    // --------------------------------------------------------

    DataCopyPadExtParams<T> padParams {
        false,
        0,
        0,
        static_cast<T>(0)
    };


    // --------------------------------------------------------
    // GM -> UB
    // --------------------------------------------------------

    DataCopyPad(
        xLocal,
        inputGMX[offset],
        copyParams,
        padParams);


    // CopyIn 完成
    inputQueueX.EnQue(
        xLocal);
}


// ============================================================
// Compute
//
// output = input_x * input_x
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 获取输入
    // --------------------------------------------------------

    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();


    // --------------------------------------------------------
    // 申请输出 LocalTensor
    // --------------------------------------------------------

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();


    // ========================================================
    // Square 真正的核心：
    //
    // y[i] = x[i] * x[i]
    //
    // 即：
    //
    // Square(x) = x²
    // ========================================================

    AscendC::Mul(
        yLocal,
        xLocal,
        xLocal,
        static_cast<int32_t>(
            currentNum));


    // 输出入队
    outputQueueY.EnQue(
        yLocal);


    // 输入已经计算完成
    inputQueueX.FreeTensor(
        xLocal);
}


// ============================================================
// CopyOut
//
// Local -> GM
//
// 同样使用 DataCopyPad。
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 获取输出 LocalTensor
    // --------------------------------------------------------

    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();


    int64_t offset =
        progress *
        ubLength_;


    // --------------------------------------------------------
    // 只搬出真实有效的数据 Byte 数。
    //
    // 这一步非常重要：
    //
    // 不会因为尾块为了 32B 对齐，
    // 而覆盖 output 后面的无效地址。
    // --------------------------------------------------------

    DataCopyExtParams copyParams {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };


    DataCopyPad(
        outputGMY[offset],
        yLocal,
        copyParams);


    // 释放输出 Tensor
    outputQueueY.FreeTensor(
        yLocal);
}


// ============================================================
// Process
//
// Tile 循环：
//
// CopyIn
//    ↓
// Compute
//    ↓
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    // --------------------------------------------------------
    // ceil(blockLength / ubLength)
    // --------------------------------------------------------

    int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;


    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        // ----------------------------------------------------
        // 当前这一轮默认处理完整 UB Tile
        // ----------------------------------------------------

        int64_t currentNum =
            ubLength_;


        // ----------------------------------------------------
        // 已经处理多少元素
        // ----------------------------------------------------

        int64_t processedNum =
            i *
            ubLength_;


        // ----------------------------------------------------
        // 还剩多少元素
        // ----------------------------------------------------

        int64_t remainNum =
            blockLength_ -
            processedNum;


        // ----------------------------------------------------
        // 最后一轮可能是不完整 Tail
        // ----------------------------------------------------

        if (remainNum < currentNum) {

            currentNum =
                remainNum;
        }


        // ----------------------------------------------------
        // Pipeline
        // ----------------------------------------------------

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