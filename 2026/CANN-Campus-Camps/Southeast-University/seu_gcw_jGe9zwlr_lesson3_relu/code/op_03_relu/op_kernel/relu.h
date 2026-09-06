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

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    // ------------------------------------------------------------
    // 1. 获取当前核编号及 tiling 参数
    // ------------------------------------------------------------
    uint32_t blockIdx = AscendC::GetBlockIdx();

    uint32_t totalNum =
        static_cast<uint32_t>(tilingData->totalNum);

    uint32_t blockFactor =
        static_cast<uint32_t>(tilingData->blockFactor);

    uint32_t ubFactor =
        static_cast<uint32_t>(tilingData->ubFactor);

    // ------------------------------------------------------------
    // 2. 当前核在整个 GM 中的起始位置
    // ------------------------------------------------------------
    uint32_t blockOffset = blockIdx * blockFactor;

    // ------------------------------------------------------------
    // 3. 当前核实际处理的数据量
    //
    // 普通核：
    //     blockLength = blockFactor
    //
    // 最后一个核：
    //     blockLength = totalNum - blockOffset
    // ------------------------------------------------------------
    if (blockOffset >= totalNum) {
        this->blockLength = 0;
    } else {
        uint32_t remain = totalNum - blockOffset;

        this->blockLength =
            remain > blockFactor ? blockFactor : remain;
    }

    // ------------------------------------------------------------
    // 4. 一个 Tile 最大处理 ubFactor 个元素
    // ------------------------------------------------------------
    this->tileLength = ubFactor;

    // ------------------------------------------------------------
    // 5. 当前核需要多少次 UB 循环
    //
    // ceil(blockLength / tileLength)
    // ------------------------------------------------------------
    if (this->blockLength == 0 || this->tileLength == 0) {
        this->tileNum = 0;
    } else {
        this->tileNum =
            (this->blockLength + this->tileLength - 1) /
            this->tileLength;
    }

    // ------------------------------------------------------------
    // 6. 设置当前核对应的 GM 起始地址
    // ------------------------------------------------------------
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        this->blockLength);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        this->blockLength);

    // ------------------------------------------------------------
    // 7. UB 队列空间
    //
    // 每次最多处理 tileLength 个元素
    // ------------------------------------------------------------
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        this->tileLength * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        this->tileLength * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    uint32_t offset =
        static_cast<uint32_t>(progress) * this->tileLength;

    AscendC::DataCopy(
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

    uint32_t offset =
        static_cast<uint32_t>(progress) * this->tileLength;

    AscendC::DataCopy(
        outputGMY[offset],
        yLocal,
        currentNum);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (uint32_t i = 0; i < this->tileNum; i++) {
        uint32_t offset = i * this->tileLength;

        // 理论上 tileNum 已经保证不会发生，
        // 这里保留判断更加安全
        if (offset >= this->blockLength) {
            break;
        }

        uint32_t remain =
            this->blockLength - offset;

        uint32_t currentNum =
            remain > this->tileLength
                ? this->tileLength
                : remain;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H