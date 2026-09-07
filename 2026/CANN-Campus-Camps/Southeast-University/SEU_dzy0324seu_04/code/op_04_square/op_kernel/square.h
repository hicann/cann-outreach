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
    __aicore__ inline Square(){};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
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
    TPipe pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    int64_t blockLength_;
    int64_t ubLength_;
    int64_t alignNum_;
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SquareTilingData* tilingData)
{
    const int64_t blockOffset =
        static_cast<int64_t>(tilingData->blockFactor) *
        AscendC::GetBlockIdx();

    const int64_t remain =
        static_cast<int64_t>(tilingData->totalNum) -
        blockOffset;

    const int64_t maxBlockLength =
        static_cast<int64_t>(tilingData->blockFactor);

    // 最后一核实际数据量可能小于 blockFactor。
    blockLength_ =
        remain < maxBlockLength ?
        remain :
        maxBlockLength;

    ubLength_ =
        static_cast<int64_t>(tilingData->ubFactor);

    // float32 每个 32B 包含 8 个元素；
    // float16 每个 32B 包含 16 个元素。
    alignNum_ = 32 / sizeof(T);

    inputGm_.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGm_.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    const uint32_t ubBufferBytes =
        static_cast<uint32_t>(
            ubLength_ * sizeof(T));

    // 输入和输出均使用 Double Buffer。
    pipe_.InitBuffer(
        inputQueue_,
        BUFFER_NUM,
        ubBufferBytes);

    pipe_.InitBuffer(
        outputQueue_,
        BUFFER_NUM,
        ubBufferBytes);
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> inputLocal =
        inputQueue_.AllocTensor<T>();

    // UB 中的计算长度向上补齐到 32B。
    const int64_t alignedNum =
        (currentNum + alignNum_ - 1) /
        alignNum_ *
        alignNum_;

    DataCopyExtParams copyParams = {
        1,
        static_cast<uint32_t>(
            currentNum * sizeof(T)),
        0,
        0,
        0
    };

    // 尾部不足 32B 的部分用 0 补齐，
    // 不从输入 GM 读取越界数据。
    DataCopyPadExtParams<T> padParams = {
        true,
        0,
        static_cast<uint8_t>(
            alignedNum - currentNum),
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

    const int32_t computeNum =
        static_cast<int32_t>(
            (currentNum + alignNum_ - 1) /
            alignNum_ *
            alignNum_);

    // y = x * x
    AscendC::Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        computeNum);

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

    // 只回写真实有效元素，补齐部分不写入 GM。
    DataCopyExtParams copyParams = {
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
    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t processed =
            i * ubLength_;

        const int64_t remain =
            blockLength_ - processed;

        const int64_t currentNum =
            remain < ubLength_ ?
            remain :
            ubLength_;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H