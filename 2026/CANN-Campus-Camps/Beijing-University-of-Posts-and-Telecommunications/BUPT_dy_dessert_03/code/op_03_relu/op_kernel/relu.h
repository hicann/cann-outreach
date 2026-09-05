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
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    const int64_t blockIdx=static_cast<int64_t>(GetBlockIdx());
    const int64_t blockFactor =static_cast<int64_t>(tilingData->blockFactor);
    const int64_t totalNum =static_cast<int64_t>(tilingData->totalNum);
    const int64_t blockOffset = blockIdx * blockFactor;
    const int64_t remainingNum = totalNum - blockOffset;
    if (remainingNum <= 0) {
        blockLength_ = 0;
    } else if (remainingNum > blockFactor) {
        blockLength_ = blockFactor;
    } else {
        blockLength_ = remainingNum;
    }

    ubLength_ = static_cast<int64_t>(tilingData->ubFactor);

    // 每个核的 GlobalTensor 都从自己的 blockOffset 开始
    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + blockOffset,
        blockLength_);

    // InitBuffer 第三个参数的单位是字节。
    //
    // inputQueueX  有两个 ubLength_ 大小的 Buffer；
    // outputQueueY 有两个 ubLength_ 大小的 Buffer。
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> inputLocal =
        inputQueueX.AllocTensor<T>();

    // progress 表示当前是第几个 UB tile
    const int64_t offset = progress * ubLength_;

    // blockLen 的单位为字节。
    // DataCopyPad 能处理最后一个 tile 非32字节对齐的情况。
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPadExtParams<T> padParams{
        false,
        0,
        0,
        0
    };

    DataCopyPad(
        inputLocal,
        inputGMX[offset],
        copyParams,
        padParams);

    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> inputLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> outputLocal =
        outputQueueY.AllocTensor<T>();

    // 必须写成 AscendC::Relu，避免和当前 Relu 类重名。
    AscendC::Relu(
        outputLocal,
        inputLocal,
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
     LocalTensor<T> outputLocal =
        outputQueueY.DeQue<T>();

    const int64_t offset = progress * ubLength_;

    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };

    // UB -> GM 使用 DataCopyPad，可以精确写出 currentNum 个元素，
    // 避免尾块写越界。
    DataCopyPad(
        outputGMY[offset],
        outputLocal,
        copyParams);

    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // 空张量或不承担任务的核直接结束。
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 向上取整，得到当前核内部需要处理多少个 UB tile。
    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t processedNum = progress * ubLength_;
        const int64_t remainingNum = blockLength_ - processedNum;

        // 最后一次循环可能不足一个完整 ubLength_
        const int64_t currentNum =
            remainingNum > ubLength_
                ? ubLength_
                : remainingNum;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
