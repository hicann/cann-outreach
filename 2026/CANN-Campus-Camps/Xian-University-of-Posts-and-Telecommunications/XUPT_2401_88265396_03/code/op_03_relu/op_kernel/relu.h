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
        BUFFER_NUM>
        inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    /*
     * 当前 Core 负责的数据量。
     */
    blockLength_ =
        tilingData->blockFactor;

    /*
     * 单次 UB 处理的数据量。
     */
    ubLength_ =
        tilingData->ubFactor;

    /*
     * Kernel 侧保护。
     *
     * 原工程自带的 kernel UT 会手工构造：
     *
     * ubFactor = totalNum
     *
     * 如果直接按照该值申请 UB，会远大于正常 UB。
     *
     * Host 正常执行时 ubFactor 已经是合法值，
     * 这里主要用于保证 Kernel 自身足够健壮。
     */
    if (ubLength_ <= 0 ||
        ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    constexpr int64_t maxTileElems = 4096;

    if (ubLength_ > maxTileElems) {
        ubLength_ = maxTileElems;
    }

    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    /*
     * 当前 Core 在 GM 中的起始位置。
     */
    const int64_t gmOffset =
        blockLength_ *
        static_cast<int64_t>(GetBlockIdx());

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + gmOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + gmOffset,
        blockLength_);

    /*
     * Input / Output 均使用 Double Buffer。
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ *
            static_cast<int64_t>(sizeof(T))));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ *
            static_cast<int64_t>(sizeof(T))));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从 VECIN Queue 申请 LocalTensor。
     */
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    /*
     * GM -> UB
     */
    AscendC::DataCopy(
        xLocal,
        inputGMX[progress],
        static_cast<uint32_t>(currentNum));

    /*
     * 放入输入 Queue，等待 Compute。
     */
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    /*
     * 获取输入 Tensor。
     */
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    /*
     * 为输出申请 Tensor。
     */
    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    /*
     * y = max(0, x)
     *
     * 显式写 AscendC::Relu，
     * 避免和当前 NsRelu::Relu 类模板同名冲突。
     */
    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<uint32_t>(currentNum));

    /*
     * 输出进入 VECOUT Queue。
     */
    outputQueueY.EnQue(yLocal);

    /*
     * 输入 Tensor 已经使用完毕。
     */
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    /*
     * 从输出 Queue 获取计算结果。
     */
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    /*
     * UB -> GM
     */
    AscendC::DataCopy(
        outputGMY[progress],
        yLocal,
        static_cast<uint32_t>(currentNum));

    /*
     * 释放输出 Tensor。
     */
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    /*
     * 当前 Core 内部继续按照 ubLength_ 分 Tile。
     */
    int64_t progress = 0;

    while (progress < blockLength_) {
        int64_t currentNum =
            blockLength_ - progress;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(
            progress,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            progress,
            currentNum);

        progress += currentNum;
    }
}

} // namespace NsRelu

#endif // RELU_H