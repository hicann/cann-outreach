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


// ============================================================
// Double Buffer
// ============================================================

constexpr int32_t BUFFER_NUM = 2;


// ============================================================
// Relu Kernel 类
// ============================================================

template <typename T>
class Relu {

public:

    __aicore__ inline Relu() {}


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

    // Pipeline 管理器
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


    // 当前 Core 实际需要处理的数据长度
    int64_t blockLength_ = 0;


    // 每次 UB Tile 处理的元素数量
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
    // --------------------------------------------------------
    // 当前 Core 的编号
    // --------------------------------------------------------

    int64_t blockIdx =
        static_cast<int64_t>(
            GetBlockIdx());


    // --------------------------------------------------------
    // 当前 Core 在整个 Tensor 中的起始位置
    //
    // 比如：
    //
    // blockFactor = 2048
    //
    // Core 0:
    // offset = 0
    //
    // Core 1:
    // offset = 2048
    //
    // Core 2:
    // offset = 4096
    // --------------------------------------------------------

    int64_t blockOffset =
        blockIdx *
        tilingData->blockFactor;


    // --------------------------------------------------------
    // 计算当前 Core 剩余多少数据
    // --------------------------------------------------------

    int64_t remainLength =
        tilingData->totalNum -
        blockOffset;


    // 当前 Core 正常最多处理 blockFactor 个
    blockLength_ =
        tilingData->blockFactor;


    // 最后一个 Core 如果不足一个完整 block，
    // 只处理真实剩余长度
    if (remainLength < blockLength_) {
        blockLength_ =
            remainLength;
    }


    // --------------------------------------------------------
    // UB 每个 Tile 的长度
    // --------------------------------------------------------

    ubLength_ =
        tilingData->ubFactor;


    if (ubLength_ > blockLength_) {
        ubLength_ =
            blockLength_;
    }


    // --------------------------------------------------------
    // 绑定当前 Core 所负责的输入 GM
    // --------------------------------------------------------

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
            blockOffset,
        blockLength_);


    // --------------------------------------------------------
    // 绑定当前 Core 所负责的输出 GM
    // --------------------------------------------------------

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
            blockOffset,
        blockLength_);


    // --------------------------------------------------------
    // 初始化输入 Queue
    //
    // BUFFER_NUM = 2
    // 即 Double Buffer
    // --------------------------------------------------------

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));


    // --------------------------------------------------------
    // 初始化输出 Queue
    // --------------------------------------------------------

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ============================================================
// CopyIn
//
// GM -> UB
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 从输入 Queue 中申请一块 LocalTensor
    // --------------------------------------------------------

    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();


    // --------------------------------------------------------
    // 当前 Tile 在当前 Core 数据中的偏移
    // --------------------------------------------------------

    int64_t offset =
        progress *
        ubLength_;


    // --------------------------------------------------------
    // GM -> LocalTensor
    // --------------------------------------------------------

    DataCopy(
        xLocal,
        inputGMX[offset],
        currentNum);


    // --------------------------------------------------------
    // 入队
    //
    // 表示 CopyIn 阶段完成，
    // Compute 阶段可以读取。
    // --------------------------------------------------------

    inputQueueX.EnQue(
        xLocal);
}


// ============================================================
// Compute
//
// y = max(0, x)
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 从输入 Queue 中取出数据
    // --------------------------------------------------------

    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();


    // --------------------------------------------------------
    // 给输出申请 LocalTensor
    // --------------------------------------------------------

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();


    // ========================================================
    // 第三题真正的核心计算
    //
    // y[i] = max(0, x[i])
    // ========================================================

    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<int32_t>(
            currentNum));


    // --------------------------------------------------------
    // 输出入队
    // --------------------------------------------------------

    outputQueueY.EnQue(
        yLocal);


    // --------------------------------------------------------
    // 输入已经使用完成
    // --------------------------------------------------------

    inputQueueX.FreeTensor(
        xLocal);
}


// ============================================================
// CopyOut
//
// UB -> GM
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    // --------------------------------------------------------
    // 从输出 Queue 中取结果
    // --------------------------------------------------------

    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();


    int64_t offset =
        progress *
        ubLength_;


    // --------------------------------------------------------
    // LocalTensor -> GM
    // --------------------------------------------------------

    DataCopy(
        outputGMY[offset],
        yLocal,
        currentNum);


    // --------------------------------------------------------
    // 释放 LocalTensor
    // --------------------------------------------------------

    outputQueueY.FreeTensor(
        yLocal);
}


// ============================================================
// Process
//
// CopyIn -> Compute -> CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // --------------------------------------------------------
    // 一个 Core 一共需要多少轮 UB Tile
    //
    // ceil(blockLength / ubLength)
    // --------------------------------------------------------

    int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;


    // --------------------------------------------------------
    // 分 Tile 处理
    // --------------------------------------------------------

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        // 默认本轮处理完整 ubLength_
        int64_t currentNum =
            ubLength_;


        // ----------------------------------------------------
        // 最后一轮可能不足一个 Tile
        // ----------------------------------------------------

        int64_t processed =
            i *
            ubLength_;


        int64_t remain =
            blockLength_ -
            processed;


        if (remain < currentNum) {
            currentNum =
                remain;
        }


        // ----------------------------------------------------
        // 三阶段流水
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


} // namespace NsRelu


#endif // RELU_H