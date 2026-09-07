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


/*
 * DoubleBuffer
 *
 * 每个 Queue 中同时准备两块 Buffer。
 */
constexpr int32_t BUFFER_NUM = 2;


template <typename T>
class Relu {
public:

    __aicore__ inline Relu() {}


    /*
     * 初始化：
     *
     * 1. 获取当前 Core 负责的 GM 数据区间
     * 2. 初始化 GlobalTensor
     * 3. 初始化 Input / Output Queue
     */
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);


    /*
     * Kernel 主处理流程
     */
    __aicore__ inline void Process();


private:

    /*
     * GM -> UB
     */
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);


    /*
     * UB -> GM
     */
    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);


    /*
     * UB 内完成 ReLU
     */
    __aicore__ inline void Compute(
        int64_t currentNum);


private:

    /*
     * Pipeline 管理对象
     */
    TPipe pipe;


    /*
     * 输入 Queue：
     *
     * GM -> VECIN
     */
    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inputQueueX;


    /*
     * 输出 Queue：
     *
     * VECOUT -> GM
     */
    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;


    /*
     * Global Memory Tensor
     */
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;


    /*
     * 当前 Core 需要处理的数据量
     */
    int64_t blockLength_ = 0;


    /*
     * 每次搬入 UB 的元素数量
     */
    int64_t ubLength_ = 0;
};


/*
 * ================================================================
 * Init
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    /*
     * 当前 Core 的编号
     */
    const int64_t blockIdx =
        static_cast<int64_t>(
            GetBlockIdx());


    /*
     * 当前 Core 对应 GM 中的起始位置。
     *
     * Core0:
     *   offset = 0
     *
     * Core1:
     *   offset = blockFactor
     *
     * Core2:
     *   offset = 2 * blockFactor
     *
     * ...
     */
    const int64_t blockOffset =
        blockIdx *
        tilingData->blockFactor;


    /*
     * 理论上当前 Core 最多处理 blockFactor。
     *
     * 同时考虑最后一个 Core 的尾块情况。
     */
    const int64_t remainNum =
        tilingData->totalNum -
        blockOffset;


    if (remainNum <= 0) {

        blockLength_ = 0;

    } else if (
        remainNum <
        tilingData->blockFactor) {

        blockLength_ =
            remainNum;

    } else {

        blockLength_ =
            tilingData->blockFactor;
    }


    /*
     * UB 每个 tile 的长度
     */
    ubLength_ =
        tilingData->ubFactor;


    /*
     * 当前 Core 的输入 GM 区域
     */
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
            blockOffset,
        blockLength_);


    /*
     * 当前 Core 的输出 GM 区域
     */
    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
            blockOffset,
        blockLength_);


    /*
     * ============================================================
     * DoubleBuffer
     * ============================================================
     *
     * Input Queue:
     *   2 × ubLength
     *
     * Output Queue:
     *   2 × ubLength
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ *
            sizeof(T));


    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ *
            sizeof(T));
}


/*
 * ================================================================
 * CopyIn
 *
 * Global Memory -> LocalTensor
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从 Input Queue 申请一块 LocalTensor
     */
    LocalTensor<T> xLocal =
        inputQueueX
            .AllocTensor<T>();


    /*
     * 当前 tile 在本 Core 数据中的 offset
     */
    const int64_t offset =
        progress *
        ubLength_;


    /*
     * GM -> UB
     */
    DataCopy(
        xLocal,
        inputGMX[offset],
        static_cast<uint32_t>(
            currentNum));


    /*
     * 放入 Queue，
     * 供 Compute 阶段使用。
     */
    inputQueueX.EnQue(
        xLocal);
}


/*
 * ================================================================
 * Compute
 *
 * y = max(0, x)
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    /*
     * 从 Input Queue 中取出输入 Tensor。
     */
    LocalTensor<T> xLocal =
        inputQueueX
            .DeQue<T>();


    /*
     * 从 Output Queue 中申请输出 Tensor。
     */
    LocalTensor<T> yLocal =
        outputQueueY
            .AllocTensor<T>();


    /*
     * ============================================================
     * ReLU
     *
     * y[i] = max(0, x[i])
     * ============================================================
     */
    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<uint32_t>(
            currentNum));


    /*
     * 输出 Tensor 入队，
     * 等待 CopyOut。
     */
    outputQueueY.EnQue(
        yLocal);


    /*
     * 输入 Tensor 已经使用完成，
     * 释放 Buffer。
     */
    inputQueueX.FreeTensor(
        xLocal);
}


/*
 * ================================================================
 * CopyOut
 *
 * LocalTensor -> Global Memory
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从输出 Queue 中取得计算结果。
     */
    LocalTensor<T> yLocal =
        outputQueueY
            .DeQue<T>();


    /*
     * 当前 tile 在本 Core 数据中的 offset。
     */
    const int64_t offset =
        progress *
        ubLength_;


    /*
     * UB -> GM
     */
    DataCopy(
        outputGMY[offset],
        yLocal,
        static_cast<uint32_t>(
            currentNum));


    /*
     * 搬出完成，
     * 释放输出 LocalTensor。
     */
    outputQueueY.FreeTensor(
        yLocal);
}


/*
 * ================================================================
 * Process
 *
 * 完整流水：
 *
 * CopyIn
 *    ↓
 * Compute
 *    ↓
 * CopyOut
 *
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    /*
     * 防止异常 Tiling 数据。
     */
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {

        return;
    }


    /*
     * 当前 Core 需要多少次 UB 循环。
     *
     * 等价于：
     *
     * ceil(blockLength / ubLength)
     */
    const int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;


    /*
     * ============================================================
     * Tile Pipeline
     * ============================================================
     */
    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        /*
         * 已经处理的元素数量
         */
        const int64_t processedNum =
            i *
            ubLength_;


        /*
         * 剩余数据
         */
        const int64_t remainNum =
            blockLength_ -
            processedNum;


        /*
         * 正常 Tile:
         *
         * currentNum = ubLength_
         *
         * 最后一个 Tile:
         *
         * currentNum = remainNum
         */
        int64_t currentNum;

        if (remainNum <
            ubLength_) {

            currentNum =
                remainNum;

        } else {

            currentNum =
                ubLength_;
        }


        /*
         * GM -> UB
         */
        CopyIn(
            i,
            currentNum);


        /*
         * ReLU
         */
        Compute(
            currentNum);


        /*
         * UB -> GM
         */
        CopyOut(
            i,
            currentNum);
    }
}


} // namespace NsRelu

#endif // RELU_H