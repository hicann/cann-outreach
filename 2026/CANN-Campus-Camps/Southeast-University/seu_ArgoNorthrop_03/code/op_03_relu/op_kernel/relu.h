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

    const ReluTilingData* tilingDataPtr_ = nullptr;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ========== kernel 实现 ==========
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // Pipe buffer
    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, sizeof(T) * tilingData->ubFactor);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, sizeof(T) * tilingData->ubFactor);

    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    // 存 tiling 指针（Process 里要读 totalNum 算最后一核真实长度）
    tilingDataPtr_ = tilingData;

    blockLength_ = tilingData->blockFactor;
    ubLength_    = tilingData->ubFactor;
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    int64_t coreOffset = blockLength_ * (int64_t)AscendC::GetBlockIdx();
    int64_t offset = coreOffset + progress * ubLength_;
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[offset], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t coreOffset = blockLength_ * (int64_t)AscendC::GetBlockIdx();
    int64_t offset = coreOffset + progress * ubLength_;
    AscendC::DataCopy(outputGMY[offset], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t coreIdx = (int64_t)AscendC::GetBlockIdx();
    int64_t totalNum = tilingDataPtr_->totalNum;

    // 每核真实长度 = min(blockFactor, totalNum - blockFactor * coreIdx)
    int64_t coreOffset = blockLength_ * coreIdx;
    if (coreOffset >= totalNum) return;  // 空核保护
    int64_t coreLen = (coreOffset + blockLength_ > totalNum)
                      ? (totalNum - coreOffset)
                      : blockLength_;

    int64_t ubLoopCount = (coreLen + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < ubLoopCount; ++i) {
        int64_t currentNum = (i == ubLoopCount - 1)
                               ? (coreLen - i * ubLength_)
                               : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
