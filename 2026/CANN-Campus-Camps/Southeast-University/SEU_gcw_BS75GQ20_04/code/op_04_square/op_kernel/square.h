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

// 单Buffer在当前轻量Square计算中更快
constexpr int32_t BUFFER_NUM = 1;

// GM基础搬运对齐字节数
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
        const SquareTilingData* tilingData,
        TPipe* pipe);

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
    // TPipe由Kernel入口创建，减少类内Scalar开销
    TPipe* pipe_ = nullptr;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前核实际处理的元素数量
    int64_t blockLength_ = 0;

    // 单次UB处理的最大元素数量
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData,
    TPipe* pipe)
{
    pipe_ = pipe;

    // 当前核在整个张量中的起始元素偏移
    int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx())
        * tilingData->blockFactor;

    int64_t remainingNum =
        tilingData->totalNum - blockOffset;

    if (remainingNum > tilingData->blockFactor) {
        blockLength_ =
            tilingData->blockFactor;
    } else if (remainingNum > 0) {
        blockLength_ =
            remainingNum;
    } else {
        blockLength_ = 0;
    }

    ubLength_ =
        tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output + blockOffset,
        blockLength_);

    pipe_->InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe_->InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    int64_t currentBytes =
        currentNum * sizeof(T);

    if ((currentBytes % ALIGN_BYTES) == 0) {
        DataCopy(
            inputLocal,
            inputGMX[offset],
            currentNum);
    } else {
        // 处理不足32字节的输入尾块
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

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        currentNum);

    outputQueueY.EnQue<T>(outputLocal);

    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    int64_t currentBytes =
        currentNum * sizeof(T);

    if ((currentBytes % ALIGN_BYTES) == 0) {
        DataCopy(
            outputGMY[offset],
            outputLocal,
            currentNum);
    } else {
        // 只向GM写回有效字节，防止尾部越界
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

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    // 常见快速路径：当前核的数据一次放入UB
    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }

    // 超大张量才进入多Tile路径
    int64_t offset = 0;

    while (offset < blockLength_) {
        int64_t remainingNum =
            blockLength_ - offset;

        int64_t currentNum =
            remainingNum > ubLength_
                ? ubLength_
                : remainingNum;

        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);

        offset += currentNum;
    }
}

} // namespace NsSquare

#endif // SQUARE_H