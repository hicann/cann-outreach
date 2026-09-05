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

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
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

    const SquareTilingData* tilingDataPtr_ = nullptr;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ========== kernel 实现 ==========
template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, sizeof(T) * tilingData->ubFactor);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, sizeof(T) * tilingData->ubFactor);

    inputGMX.SetGlobalBuffer((__gm__ T*)input_x);
    outputGMY.SetGlobalBuffer((__gm__ T*)output);

    tilingDataPtr_ = tilingData;
    blockLength_   = tilingData->blockFactor;
    ubLength_      = tilingData->ubFactor;
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    int64_t coreOffset = blockLength_ * (int64_t)AscendC::GetBlockIdx();
    int64_t offset = coreOffset + progress * ubLength_;
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[offset], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // Square = x * x（Mul 已验证可用）
    AscendC::Mul(yLocal, xLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t coreOffset = blockLength_ * (int64_t)AscendC::GetBlockIdx();
    int64_t offset = coreOffset + progress * ubLength_;
    AscendC::DataCopy(outputGMY[offset], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    int64_t coreIdx  = (int64_t)AscendC::GetBlockIdx();
    int64_t totalNum = tilingDataPtr_->totalNum;

    // alignElements：fp32=8, fp16=16（32B / sizeof(T)）
    constexpr int64_t ALIGN_FP32 = 8;
    constexpr int64_t ALIGN_FP16 = 16;
    constexpr int64_t alignElements = (sizeof(T) == 4) ? ALIGN_FP32 : ALIGN_FP16;

    int64_t coreOffset = blockLength_ * coreIdx;
    if (coreOffset >= totalNum) return;

    // 每核真实长度（CeilAlign 防最后一块非对齐越界）
    int64_t coreLenRaw = (coreOffset + blockLength_ > totalNum)
                         ? (totalNum - coreOffset)
                         : blockLength_;
    int64_t coreLen = (coreLenRaw + alignElements - 1) / alignElements * alignElements;
    if (coreOffset + coreLen > totalNum) coreLen = totalNum - coreOffset;  // 保护不越界

    int64_t ubLoopCount = (coreLen + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < ubLoopCount; ++i) {
        int64_t currentNumReal = (i == ubLoopCount - 1)
                                   ? (coreLen - i * ubLength_)
                                   : ubLength_;
        // CeilAlign 后保证向量化不越界
        int64_t currentNum = (currentNumReal + alignElements - 1) / alignElements * alignElements;
        // 最后一块对齐后可能超出，截断到 ubLength_
        if (currentNum > ubLength_) currentNum = ubLength_;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
