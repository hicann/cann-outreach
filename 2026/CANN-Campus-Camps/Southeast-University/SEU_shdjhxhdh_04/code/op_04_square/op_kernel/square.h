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

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);


private:

    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;

    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;

    int64_t ubLength_ = 0;
};


/*
 * ============================================================
 * Init
 * ============================================================
 */
template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    /*
     * 当前核编号
     */
    int64_t blockIdx =
        GetBlockIdx();


    /*
     * 设置 GM Tensor
     */
    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            input_x));

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            output));


    /*
     * 获取 Tiling 参数
     */
    int64_t totalNum =
        tilingData->totalNum;

    int64_t blockFactor =
        tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;


    /*
     * 当前核在整个输入中的起始位置
     */
    int64_t blockStart =
        blockIdx * blockFactor;


    /*
     * 当前核实际需要处理的数据量
     */
    if (blockStart >= totalNum) {

        blockLength_ = 0;

    } else {

        int64_t remain =
            totalNum - blockStart;

        blockLength_ =
            (remain < blockFactor)
                ? remain
                : blockFactor;
    }


    /*
     * GM 指针移动到当前核负责区域
     */
    inputGMX =
        inputGMX[blockStart];

    outputGMY =
        outputGMY[blockStart];


    /*
     * 输入双缓冲
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));


    /*
     * 输出双缓冲
     */
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


/*
 * ============================================================
 * CopyIn
 * GM -> UB
 * ============================================================
 */
template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 获取输入 LocalTensor
     */
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();


    /*
     * 当前 tile 在本核数据中的偏移
     */
    int64_t offset =
        progress * ubLength_;


    /*
     * 实际需要搬运的字节数
     */
    uint32_t copyBytes =
        static_cast<uint32_t>(
            currentNum * sizeof(T));


    /*
     * GM -> UB 参数
     */
    DataCopyExtParams copyParams = {
        1,
        copyBytes,
        0,
        0,
        0
    };


    /*
     * 右侧补齐到 32 Byte
     *
     * 例如 float16：
     *
     * currentNum = 1
     *
     * 实际：
     *     2 Byte
     *
     * 需要：
     *     32 Byte
     *
     * rightPadding = 15
     */
    constexpr int64_t ALIGN_BYTES = 32;

    int64_t currentBytes =
        currentNum * sizeof(T);

    int64_t alignedBytes =
        ((currentBytes +
          ALIGN_BYTES - 1) /
         ALIGN_BYTES) *
        ALIGN_BYTES;

    int64_t paddingBytes =
        alignedBytes -
        currentBytes;

    uint8_t rightPadding =
        static_cast<uint8_t>(
            paddingBytes / sizeof(T));


    /*
     * DataCopyPad 参数
     */
    DataCopyPadExtParams<T> padParams = {
        true,
        0,
        rightPadding,
        static_cast<T>(0)
    };


    /*
     * GM -> UB
     */
    DataCopyPad(
        inputLocal,
        inputGMX[offset],
        copyParams,
        padParams);


    /*
     * 入队
     */
    inputQueueX.EnQue(
        inputLocal);
}


/*
 * ============================================================
 * Compute
 *
 * y = x * x
 * ============================================================
 */
template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    /*
     * 获取输入
     */
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();


    /*
     * 获取输出
     */
    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();


    /*
     * Square:
     *
     * y = x * x
     */
    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        currentNum);


    /*
     * 输出入队
     */
    outputQueueY.EnQue(
        outputLocal);


    /*
     * 释放输入
     */
    inputQueueX.FreeTensor(
        inputLocal);
}


/*
 * ============================================================
 * CopyOut
 * UB -> GM
 * ============================================================
 */
template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 获取输出
     */
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();


    /*
     * 当前 tile 的 GM 偏移
     */
    int64_t offset =
        progress * ubLength_;


    /*
     * 只写有效数据
     */
    DataCopyExtParams copyParams = {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };


    /*
     * UB -> GM
     */
    DataCopyPad(
        outputGMY[offset],
        outputLocal,
        copyParams);


    /*
     * 释放输出
     */
    outputQueueY.FreeTensor(
        outputLocal);
}


/*
 * ============================================================
 * Process
 * ============================================================
 */
template <typename T>
__aicore__ inline void Square<T>::Process()
{
    /*
     * 当前核没有数据
     */
    if (blockLength_ <= 0) {
        return;
    }


    /*
     * UB tile 数量
     */
    int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;


    for (int64_t progress = 0;
         progress < loopCount;
         ++progress) {

        /*
         * 当前 tile 起点
         */
        int64_t offset =
            progress * ubLength_;


        /*
         * 剩余元素
         */
        int64_t remain =
            blockLength_ - offset;


        /*
         * 当前 tile 实际元素数量
         */
        int64_t currentNum =
            (remain < ubLength_)
                ? remain
                : ubLength_;


        /*
         * GM -> UB
         */
        CopyIn(
            progress,
            currentNum);


        /*
         * y = x * x
         */
        Compute(
            currentNum);


        /*
         * UB -> GM
         */
        CopyOut(
            progress,
            currentNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H