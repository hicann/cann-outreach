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
constexpr uint32_t VECTOR_BYTES_PER_REPEAT = 256;
constexpr uint32_t MAX_VECTOR_REPEAT = 255;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);

private:
    TPipe pipe;
    // Queue depth is one because there is no consecutive EnQue. Double
    // buffering is configured by the num argument of InitBuffer below.
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t core = GetBlockIdx();
    const int64_t offset = core * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining > 0 ? (remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor) : 0;
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ == 0) {
        return;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(uint32_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    // The level-2 Relu API accepts at most 255 vector repeats per call.
    // A 64 KiB UB tile is slightly larger than that limit, so split only the
    // vector calculation while keeping one large GM transfer per tile.
    constexpr uint32_t maxElementsPerCall =
        VECTOR_BYTES_PER_REPEAT * MAX_VECTOR_REPEAT / sizeof(T);
    for (uint32_t offset = 0; offset < currentNum; offset += maxElementsPerCall) {
        const uint32_t remaining = currentNum - offset;
        const uint32_t computeNum = remaining < maxElementsPerCall ? remaining : maxElementsPerCall;
        AscendC::Relu(outputLocal[offset], inputLocal[offset], computeNum);
    }
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        const int64_t remaining = blockLength_ - progress;
        const uint32_t currentNum = static_cast<uint32_t>(remaining < ubLength_ ? remaining : ubLength_);
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
