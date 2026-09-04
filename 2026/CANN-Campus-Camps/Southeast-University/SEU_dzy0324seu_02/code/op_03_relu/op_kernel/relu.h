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


    int64_t blockLength_ = 0;

    int64_t ubLength_ = 0;
};


// ========================================================
// Init
// ========================================================

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    /*
     * 每个 Core 处理的数据数量
     */
    blockLength_ =
        tilingData->blockFactor;


    /*
     * 每次 UB 循环处理的数据数量
     */
    ubLength_ =
        tilingData->ubFactor;


    /*
     * 当前 Core 在 GM 中的偏移
     *
     * 当前方案 blockDim = 1，
     * 因此 Core0 offset = 0。
     */
    int64_t gmOffset =
        static_cast<int64_t>(
            GetBlockIdx()) *
        blockLength_;


    /*
     * 初始化输入 GlobalTensor
     */
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + gmOffset,
        blockLength_);


    /*
     * 初始化输出 GlobalTensor
     */
    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + gmOffset,
        blockLength_);


    /*
     * 输入 Double Buffer
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));


    /*
     * 输出 Double Buffer
     */
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ========================================================
// CopyIn
//
// Global Memory -> Unified Buffer
// ========================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();


    /*
     * progress 表示元素偏移，不是 Tile 编号
     */
    DataCopy(
        inputLocal,
        inputGMX[progress],
        currentNum);


    inputQueueX.EnQue(
        inputLocal);
}


// ========================================================
// Compute
//
// y = max(0, x)
// ========================================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();


    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();


    /*
     * 必须带 AscendC:: 前缀。
     *
     * 因为当前 Kernel 类本身也叫 Relu。
     */
    AscendC::Relu(
        outputLocal,
        inputLocal,
        static_cast<int32_t>(
            currentNum));


    outputQueueY.EnQue(
        outputLocal);


    inputQueueX.FreeTensor(
        inputLocal);
}


// ========================================================
// CopyOut
//
// Unified Buffer -> Global Memory
// ========================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();


    DataCopy(
        outputGMY[progress],
        outputLocal,
        currentNum);


    outputQueueY.FreeTensor(
        outputLocal);
}


// ========================================================
// Process
// ========================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    /*
     * progress 直接表示已经处理的元素数量。
     */
    int64_t progress = 0;


    while (progress < blockLength_)
    {
        /*
         * 默认每次处理一个完整 UB Tile。
         */
        int64_t currentNum =
            ubLength_;


        /*
         * 最后一次不足一个 Tile 时，
         * 只处理剩余元素。
         */
        if (progress + currentNum >
            blockLength_)
        {
            currentNum =
                blockLength_ -
                progress;
        }


        /*
         * GM -> UB
         */
        CopyIn(
            progress,
            currentNum);


        /*
         * ReLU计算
         */
        Compute(
            currentNum);


        /*
         * UB -> GM
         */
        CopyOut(
            progress,
            currentNum);


        /*
         * 移动到下一段
         */
        progress +=
            currentNum;
    }
}


} // namespace NsRelu

#endif // RELU_H