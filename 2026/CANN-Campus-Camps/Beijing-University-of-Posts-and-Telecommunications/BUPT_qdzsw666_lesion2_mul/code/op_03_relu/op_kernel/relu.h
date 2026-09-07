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
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前核实际需要处理的数据量
    int64_t blockLength_ = 0;

    // 每个Tile最多处理的数据量
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t totalNum =
        static_cast<int64_t>(tilingData->totalNum);

    const int64_t blockFactor =
        static_cast<int64_t>(tilingData->blockFactor);

    ubLength_ =
        static_cast<int64_t>(tilingData->ubFactor);

    // 当前核负责数据的起始位置
    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) *
        blockFactor;

    /*
     * 最后一个核的数据量可能不足blockFactor。
     * 如果当前核起点已经超过totalNum，则不处理数据。
     */
    if (blockOffset >= totalNum) {
        blockLength_ = 0;
    } else {
        const int64_t remainNum =
            totalNum - blockOffset;

        blockLength_ =
            remainNum < blockFactor
                ? remainNum
                : blockFactor;
    }

    // 设置当前核对应的GM区域
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    // 为输入、输出队列分配UB空间
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
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    const int64_t offset =
        progress * ubLength_;

    // GM -> UB
    DataCopy(
        xLocal,
        inputGMX[offset],
        currentNum);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // y = max(x, 0)
    AscendC::Relu(
        yLocal,
        xLocal,
        currentNum);

    outputQueueY.EnQue<T>(yLocal);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    const int64_t offset =
        progress * ubLength_;

    // UB -> GM
    DataCopy(
        outputGMY[offset],
        yLocal,
        currentNum);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 向上取整得到当前核需要处理的Tile数量
    const int64_t tileNum =
        (blockLength_ + ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0; i < tileNum; ++i) {
        const int64_t offset =
            i * ubLength_;

        const int64_t remainNum =
            blockLength_ - offset;

        // 最后一个Tile可能不足ubLength_
        const int64_t currentNum =
            remainNum < ubLength_
                ? remainNum
                : ubLength_;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H