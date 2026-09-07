/*!
 * \file square.h
 * \brief Square 算子 Kernel 类定义
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
constexpr int64_t DATA_BLOCK_BYTES = 32;

template <typename T>
class Square {
public:
    __aicore__ inline Square() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
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
    // UB 队列管理
    TPipe pipe_;

    // 输入和输出均使用双缓冲
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue_;

    // 当前核对应的 GM 张量
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    // 当前核实际处理的元素数
    int64_t blockLength_ = 0;

    // 每核最多处理的元素数
    int64_t blockFactor_ = 0;

    // 每个 UB tile 最多处理的元素数
    int64_t ubLength_ = 0;

    // 32 字节对应的元素数
    int64_t alignNum_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SquareTilingData* tilingData)
{
    blockFactor_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // float32: 8 elements
    // float16: 16 elements
    alignNum_ =
        DATA_BLOCK_BYTES /
        static_cast<int64_t>(sizeof(T));

    // 当前核在整个输入张量中的起始元素下标
    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) *
        blockFactor_;

    // 当前核还剩多少有效元素
    const int64_t remaining =
        tilingData->totalNum - blockOffset;

    // 最后一个核可能不足 blockFactor
    blockLength_ =
        remaining < blockFactor_
            ? remaining
            : blockFactor_;

    // 绑定当前核负责的 GM 区间
    inputGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) +
            blockOffset,
        blockLength_);

    outputGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) +
            blockOffset,
        blockLength_);

    // 输入队列双缓冲
    pipe_.InitBuffer(
        inputQueue_,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    // 输出队列双缓冲
    pipe_.InitBuffer(
        outputQueue_,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueue_.AllocTensor<T>();

    // DataCopyPad 的 blockLen 单位是字节。
    // currentNum 可以不是 32 字节的整数倍。
    DataCopyExtParams copyParams {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPadExtParams<T> padParams {
        false,
        0,
        0,
        static_cast<T>(0)
    };

    DataCopyPad(
        inputLocal,
        inputGm_[progress * ubLength_],
        copyParams,
        padParams);

    inputQueue_.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueue_.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueue_.AllocTensor<T>();

    // Vector 计算长度向上对齐到 32 字节。
    //
    // UB buffer 的容量为 ubLength_，而 ubLength_ 已经对齐，
    // 因此这里向上对齐不会超过已分配的 UB buffer。
    const int64_t alignedNum =
        ((currentNum + alignNum_ - 1) /
         alignNum_) *
        alignNum_;

    // y = x * x
    Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        static_cast<int32_t>(alignedNum));

    outputQueue_.EnQue(outputLocal);
    inputQueue_.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> outputLocal =
        outputQueue_.DeQue<T>();

    // 只搬出真实有效的 currentNum 个元素。
    // 即使计算长度向上对齐，也不会把对齐区域写回 GM。
    DataCopyExtParams copyParams {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(
        outputGm_[progress * ubLength_],
        outputLocal,
        copyParams);

    outputQueue_.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    // 当前核需要处理多少个 UB tile
    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) /
        ubLength_;

    for (int64_t progress = 0;
         progress < loopCount;
         ++progress) {
        const int64_t processed =
            progress * ubLength_;

        const int64_t remaining =
            blockLength_ - processed;

        // 最后一个 tile 可能不足 ubLength_
        const int64_t currentNum =
            remaining < ubLength_
                ? remaining
                : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H