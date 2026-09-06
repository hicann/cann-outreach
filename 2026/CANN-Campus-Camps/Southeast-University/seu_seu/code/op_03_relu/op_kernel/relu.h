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
 * Double Buffer
 *
 * input:
 *     buffer 0
 *     buffer 1
 *
 * output:
 *     buffer 0
 *     buffer 1
 */
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
    /*
     * Pipeline
     */
    TPipe pipe;


    /*
     * GM -> UB
     *
     * Double Buffer
     */
    TQue<
        QuePosition::VECIN,
        BUFFER_NUM> inputQueueX;


    /*
     * UB -> GM
     *
     * Double Buffer
     */
    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM> outputQueueY;


    /*
     * Global Memory Tensor
     */
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;


    /*
     * 当前 Core 实际需要处理的数据量。
     *
     * 最后一个 Core 可能少于 blockFactor。
     */
    int64_t blockLength_ = 0;


    /*
     * 一轮 UB 可以处理的最大元素数量。
     */
    int64_t ubLength_ = 0;
};


// ====================================================================
// Init
// ====================================================================
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    /*
     * 当前 Core 的全局数据起始位置：
     *
     * blockOffset =
     *     blockFactor * blockIdx
     */
    int64_t blockOffset =
        tilingData->blockFactor *
        AscendC::GetBlockIdx();


    /*
     * 剩余数据数量。
     *
     * 最后一个 Core 的数据可能不足
     * blockFactor，因此不能直接全部按照
     * blockFactor 处理。
     */
    int64_t remainderLength =
        tilingData->totalNum -
        blockOffset;


    /*
     * 当前 Core 真正需要处理的数据量。
     */
    blockLength_ =
        (remainderLength >
         tilingData->blockFactor)
            ? tilingData->blockFactor
            : remainderLength;


    /*
     * 防御：
     * 正常 Host Tiling 不会启动无数据 Core。
     */
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }


    /*
     * 单次 UB 最大处理元素数量。
     */
    ubLength_ =
        tilingData->ubFactor;


    // ================================================================
    // 设置当前 Core 对应的 GM 地址
    // ================================================================
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
            blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
            blockOffset,
        blockLength_);


    // ================================================================
    // 初始化 UB Queue
    //
    // BUFFER_NUM = 2
    //
    // 即启用 Double Buffer。
    // ================================================================
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ====================================================================
// CopyIn
//
// GM -> UB
// ====================================================================
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从 VECIN Queue 申请一块 LocalTensor。
     */
    AscendC::LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();


    /*
     * DataCopyPad 支持非 32B 整倍数的数据搬运。
     *
     * currentNum 是元素个数，
     * blockLen 的单位是 Byte。
     */
    AscendC::DataCopyParams copyParams;

    copyParams.blockCount = 1;

    copyParams.blockLen =
        static_cast<uint16_t>(
            currentNum *
            sizeof(T));

    copyParams.srcStride = 0;
    copyParams.dstStride = 0;


    /*
     * 当前 Tile 在本 Core 数据中的偏移：
     *
     * progress * ubLength_
     */
    AscendC::DataCopyPad(
        xLocal,
        inputGMX[
            progress *
            ubLength_],
        copyParams,
        {
            false,
            0,
            0,
            0
        });


    /*
     * 入队，通知 Compute 可以使用。
     */
    inputQueueX.EnQue(
        xLocal);
}


// ====================================================================
// Compute
//
// y = max(0, x)
// ====================================================================
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    /*
     * 取出已经搬入 UB 的输入 Tensor。
     */
    AscendC::LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();


    /*
     * 为输出申请 LocalTensor。
     */
    AscendC::LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();


    /*
     * ReLU:
     *
     * y[i] = max(0, x[i])
     *
     * T:
     *
     * float16 -> half
     * float32 -> float
     */
    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<int32_t>(
            currentNum));


    /*
     * 计算完成后输出入队。
     */
    outputQueueY.EnQue<T>(
        yLocal);


    /*
     * 输入已经计算完毕，
     * 释放对应 UB Tensor。
     */
    inputQueueX.FreeTensor(
        xLocal);
}


// ====================================================================
// CopyOut
//
// UB -> GM
// ====================================================================
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从 VECOUT Queue 获取计算结果。
     */
    AscendC::LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();


    /*
     * DataCopyPad：
     *
     * 可以精确搬出 currentNum 个元素，
     * 不要求 currentNum * sizeof(T)
     * 是 32B 的整数倍。
     */
    AscendC::DataCopyParams copyParams;

    copyParams.blockCount = 1;

    copyParams.blockLen =
        static_cast<uint16_t>(
            currentNum *
            sizeof(T));

    copyParams.srcStride = 0;
    copyParams.dstStride = 0;


    /*
     * UB -> GM
     */
    AscendC::DataCopyPad(
        outputGMY[
            progress *
            ubLength_],
        yLocal,
        copyParams);


    /*
     * 搬出完成后释放输出 UB Tensor。
     */
    outputQueueY.FreeTensor(
        yLocal);
}


// ====================================================================
// Process
// ====================================================================
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    /*
     * 空 Tensor / 无数据 Core。
     */
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }


    /*
     * 当前 Core 总共需要多少轮 UB 循环：
     *
     * ceil(blockLength / ubLength)
     */
    int64_t loopCount =
        (blockLength_ +
         ubLength_ -
         1) /
        ubLength_;


    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        /*
         * 最后一轮可能不足 ubLength_。
         */
        int64_t currentNum =
            (i ==
             loopCount - 1)
                ? (
                    blockLength_ -
                    ubLength_ *
                    i)
                : ubLength_;


        // ============================================================
        // Stage 1
        //
        // GM -> UB
        // ============================================================
        CopyIn(
            i,
            currentNum);


        // ============================================================
        // Stage 2
        //
        // Vector ReLU
        // ============================================================
        Compute(
            currentNum);


        // ============================================================
        // Stage 3
        //
        // UB -> GM
        // ============================================================
        CopyOut(
            i,
            currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
