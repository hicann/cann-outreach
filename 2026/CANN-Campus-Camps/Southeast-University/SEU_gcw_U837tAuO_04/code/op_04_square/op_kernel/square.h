/*!
 * \file square.h
 * \brief Square算子Kernel实现
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

// 使用单Buffer，减少UB占用和Queue调度开销
constexpr int32_t BUFFER_NUM = 1;

// Ascend C数据搬运基本对齐字节数
constexpr int64_t ALIGN_BYTES = 32;

template <typename T>
class Square {
public:
    __aicore__ inline Square()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t offset,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t offset,
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前核实际需要处理的元素数量
    int64_t blockLength_ = 0;

    // 每个Tile最多处理的元素数量
    int64_t ubLength_ = 0;
};

// ============================================================
// 初始化
// ============================================================
template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    // 当前核在整个输入Tensor中的起始元素下标
    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx())
        * tilingData->blockFactor;

    // 从当前核起始位置到Tensor末尾的剩余元素数
    const int64_t remainingNum =
        tilingData->totalNum - blockOffset;

    /*
     * 普通核心处理blockFactor个元素；
     * 最后一个核心只处理实际剩余元素。
     */
    if (remainingNum > tilingData->blockFactor) {
        blockLength_ =
            tilingData->blockFactor;
    } else {
        blockLength_ =
            remainingNum;
    }

    // 每次搬入UB并计算的最大元素数量
    ubLength_ =
        tilingData->ubFactor;

    // 将输入GlobalTensor定位到当前核心负责的数据区域
    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(input_x)
            + blockOffset,
        blockLength_);

    // 将输出GlobalTensor定位到当前核心负责的数据区域
    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(output)
            + blockOffset,
        blockLength_);

    // 为输入Queue申请一个UB Buffer
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    // 为输出Queue申请一个UB Buffer
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

// ============================================================
// 输入搬运：GM -> UB
// ============================================================
template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    const int64_t currentBytes =
        currentNum * sizeof(T);

    if (currentBytes % ALIGN_BYTES == 0) {
        /*
         * 当前数据长度满足32字节对齐，
         * 使用普通DataCopy。
         */
        DataCopy(
            inputLocal,
            inputGMX[offset],
            currentNum);
    } else {
        /*
         * 当前数据长度不满足32字节对齐。
         * 使用DataCopyPad处理最后不足32字节的数据，
         * 避免普通DataCopy读取输入Tensor范围之外的数据。
         */
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentBytes),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> padParams{
            true,
            0,
            0,
            static_cast<T>(0)
        };

        DataCopyPad(
            inputLocal,
            inputGMX[offset],
            copyParams,
            padParams);
    }

    inputQueueX.EnQue<T>(inputLocal);
}

// ============================================================
// 计算：output = input_x * input_x
// ============================================================
template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // 对有效的currentNum个元素执行逐元素平方
    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        currentNum);

    outputQueueY.EnQue<T>(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}

// ============================================================
// 输出搬运：UB -> GM
// ============================================================
template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    const int64_t currentBytes =
        currentNum * sizeof(T);

    if (currentBytes % ALIGN_BYTES == 0) {
        /*
         * 当前数据长度满足32字节对齐，
         * 使用普通DataCopy。
         */
        DataCopy(
            outputGMY[offset],
            outputLocal,
            currentNum);
    } else {
        /*
         * 当前数据长度不满足32字节对齐。
         * UB到GM方向使用三参数DataCopyPad，
         * 只向输出Tensor写入currentBytes个有效字节。
         */
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentBytes),
            0,
            0,
            0
        };

        DataCopyPad(
            outputGMY[offset],
            outputLocal,
            copyParams);
    }

    outputQueueY.FreeTensor(outputLocal);
}

// ============================================================
// 完整执行流程
// ============================================================
template <typename T>
__aicore__ inline void Square<T>::Process()
{
    /*
     * 快速路径：
     * 如果当前核负责的数据可以一次放入UB，
     * 就直接执行一次搬入、计算和搬出。
     */
    if (blockLength_ <= ubLength_) {
        CopyIn(
            0,
            blockLength_);

        Compute(
            blockLength_);

        CopyOut(
            0,
            blockLength_);

        return;
    }

    /*
     * 多Tile路径：
     * 当当前核负责的数据超过UB容量时，
     * 分成多个Tile依次处理。
     *
     * offset直接递增，避免每次循环重新执行
     * progress * ubLength_。
     */
    int64_t offset = 0;

    while (offset < blockLength_) {
        const int64_t remainingNum =
            blockLength_ - offset;

        const int64_t currentNum =
            remainingNum > ubLength_
                ? ubLength_
                : remainingNum;

        CopyIn(
            offset,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            offset,
            currentNum);

        offset += currentNum;
    }
}

} // namespace NsSquare

#endif // SQUARE_H