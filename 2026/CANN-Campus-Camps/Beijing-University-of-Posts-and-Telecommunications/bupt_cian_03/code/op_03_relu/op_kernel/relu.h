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

// 开启 DoubleBuffer
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
    // =========================================================
    // Pipeline
    // =========================================================

    TPipe pipe;

    // 输入 DoubleBuffer
    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inputQueueX;

    // 输出 DoubleBuffer
    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;

    // =========================================================
    // Global Memory Tensor
    // =========================================================

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前 Core 实际负责的元素数量
    // 最后一个 Core 可能小于 blockFactor
    int64_t blockLength_ = 0;

    // 单个 DoubleBuffer Slot 的 Tile 大小
    int64_t ubLength_ = 0;
};


// =============================================================
// Init
// =============================================================

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t totalNum =
        tilingData->totalNum;

    const int64_t blockFactor =
        tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;

    // 当前 AIV Core 编号
    const int64_t blockIdx =
        static_cast<int64_t>(
            AscendC::GetBlockIdx());

    // 当前核负责的数据在整个 Tensor 中的起始位置
    const int64_t blockOffset =
        blockIdx * blockFactor;

    // =========================================================
    // 计算当前核真正需要处理的数据量
    // =========================================================

    if (blockOffset >= totalNum ||
        blockFactor <= 0) {

        blockLength_ = 0;

    } else {

        const int64_t remainNum =
            totalNum - blockOffset;

        blockLength_ =
            remainNum < blockFactor
                ? remainNum
                : blockFactor;
    }

    // =========================================================
    // 设置 GlobalTensor
    // =========================================================

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    // =========================================================
    // UB Buffer
    // =========================================================
    //
    // inputQueueX:
    //
    //   buffer 0
    //   buffer 1
    //
    // outputQueueY:
    //
    //   buffer 0
    //   buffer 1
    //
    // BUFFER_NUM = 2 开启 DoubleBuffer。
    //
    // ubLength_ 已经在 Host Tiling 中根据：
    //
    //     UB Size / 4
    //
    // 计算。
    // =========================================================

    const uint32_t bufferBytes =
        static_cast<uint32_t>(
            ubLength_
            * static_cast<int64_t>(sizeof(T)));

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        bufferBytes);

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        bufferBytes);
}


// =============================================================
// CopyIn
// =============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    // 从输入队列申请一块 LocalTensor
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    const uint32_t copyBytes =
        static_cast<uint32_t>(
            currentNum
            * static_cast<int64_t>(sizeof(T)));

    // =========================================================
    // DataCopy 参数
    // =========================================================
    //
    // 使用 DataCopyPad 的原因：
    //
    // 最后一个 Tile 未必满足 32 Byte 对齐。
    //
    // 例如 FP32:
    //
    // currentNum = 13
    // bytes = 52
    //
    // 52 不是 32 的整数倍。
    //
    // DataCopyPad 可以正确处理这种尾块。
    // =========================================================

    DataCopyExtParams copyParams;

    copyParams.blockCount = 1;
    copyParams.blockLen = copyBytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams;

    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue =
        static_cast<T>(0);

    // =========================================================
    // GM -> UB
    // =========================================================

    DataCopyPad(
        xLocal,
        inputGMX[progress],
        copyParams,
        padParams);

    // 放入 VECIN 队列
    inputQueueX.EnQue(
        xLocal);
}


// =============================================================
// Compute
// =============================================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    // 从输入队列取数据
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    // 为输出申请 LocalTensor
    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // =========================================================
    // ReLU
    //
    // y = max(0, x)
    // =========================================================

    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<int32_t>(
            currentNum));

    // 放入输出队列
    outputQueueY.EnQue(
        yLocal);

    // 输入 Tensor 已完成计算，可以释放
    inputQueueX.FreeTensor(
        xLocal);
}


// =============================================================
// CopyOut
// =============================================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    // 从 VECOUT 队列取计算结果
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    const uint32_t copyBytes =
        static_cast<uint32_t>(
            currentNum
            * static_cast<int64_t>(sizeof(T)));

    DataCopyExtParams copyParams;

    copyParams.blockCount = 1;
    copyParams.blockLen = copyBytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    // =========================================================
    // UB -> GM
    // =========================================================

    DataCopyPad(
        outputGMY[progress],
        yLocal,
        copyParams);

    // 当前输出 Buffer 可以重新使用
    outputQueueY.FreeTensor(
        yLocal);
}


// =============================================================
// Process
// =============================================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 空 Tensor 或异常 Tiling
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {

        return;
    }

    // =========================================================
    // 每个 Core 内进一步进行 UB Tiling
    // =========================================================

    const int64_t loopCount =
        (blockLength_
            + ubLength_
            - 1)
        / ubLength_;

    // =========================================================
    // Pipeline:
    //
    // CopyIn -> Compute -> CopyOut
    //
    // 由于：
    //
    // TQue<..., BUFFER_NUM>
    //
    // 中 BUFFER_NUM = 2，
    // 输入/输出 Queue 各自拥有两个 Buffer Slot。
    //
    // 因此不同 Tile 的 MTE 与 Vector 指令能够形成流水，
    // 降低数据搬运和计算串行等待。
    // =========================================================

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        // 当前 Tile 在本 Core 内的起始下标
        const int64_t progress =
            i * ubLength_;

        // 剩余元素
        const int64_t remainNum =
            blockLength_
            - progress;

        // 最后一个 Tile 可能不足 ubLength_
        const int64_t currentNum =
            remainNum < ubLength_
                ? remainNum
                : ubLength_;

        CopyIn(
            progress,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            progress,
            currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H