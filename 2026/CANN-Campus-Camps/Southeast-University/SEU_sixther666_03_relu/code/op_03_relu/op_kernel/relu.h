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
constexpr int32_t QUEUE_DEPTH = 1;
constexpr uint32_t FAST_BLOCK_LENGTH = 1024;
constexpr uint32_t FAST_TILE_LENGTH = 512;

template <typename T, bool FAST_FIXED_SHAPE>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentNum);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);

private:
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueueX;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    uint32_t blockLength_ = 0;
    uint32_t ubLength_ = 0;
};

template <typename T, bool FAST_FIXED_SHAPE>
__aicore__ inline void Relu<T, FAST_FIXED_SHAPE>::Init(
    GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData, TPipe* pipe)
{
    if constexpr (FAST_FIXED_SHAPE) {
        const uint32_t blockOffset = GetBlockIdx() * FAST_BLOCK_LENGTH;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, FAST_BLOCK_LENGTH);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, FAST_BLOCK_LENGTH);
        pipe->InitBuffer(inputQueueX, BUFFER_NUM, FAST_TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outputQueueY, BUFFER_NUM, FAST_TILE_LENGTH * sizeof(T));
    } else {
        const uint32_t blockFactor = static_cast<uint32_t>(tilingData->blockFactor);
        const uint32_t blockOffset = GetBlockIdx() * blockFactor;
        const uint32_t totalNum = static_cast<uint32_t>(tilingData->totalNum);
        const uint32_t remainingNum = blockOffset < totalNum ? totalNum - blockOffset : 0;
        blockLength_ = remainingNum < blockFactor ? remainingNum : blockFactor;
        ubLength_ = static_cast<uint32_t>(tilingData->ubFactor);
        inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
        if (blockLength_ > 0 && ubLength_ > 0) {
            pipe->InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
            pipe->InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
        }
    }
}

template <typename T, bool FAST_FIXED_SHAPE>
__aicore__ inline void Relu<T, FAST_FIXED_SHAPE>::CopyIn(uint32_t offset, uint32_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[offset], currentNum);
    inputQueueX.EnQue(inputLocal);
}

template <typename T, bool FAST_FIXED_SHAPE>
__aicore__ inline void Relu<T, FAST_FIXED_SHAPE>::Compute(uint32_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(outputLocal, inputLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T, bool FAST_FIXED_SHAPE>
__aicore__ inline void Relu<T, FAST_FIXED_SHAPE>::CopyOut(uint32_t offset, uint32_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[offset], outputLocal, currentNum);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T, bool FAST_FIXED_SHAPE>
__aicore__ inline void Relu<T, FAST_FIXED_SHAPE>::Process()
{
    if constexpr (FAST_FIXED_SHAPE) {
        CopyIn(0, FAST_TILE_LENGTH);
        Compute(FAST_TILE_LENGTH);
        CopyOut(0, FAST_TILE_LENGTH);
        CopyIn(FAST_TILE_LENGTH, FAST_TILE_LENGTH);
        Compute(FAST_TILE_LENGTH);
        CopyOut(FAST_TILE_LENGTH, FAST_TILE_LENGTH);
        return;
    }

    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }

    uint32_t offset = 0;
    while (offset < blockLength_) {
        const uint32_t remainingNum = blockLength_ - offset;
        const uint32_t currentNum = remainingNum < ubLength_ ? remainingNum : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsRelu
#endif // RELU_H
