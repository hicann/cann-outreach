/*!
 * \file truncate_mod.h
 * \brief TruncateMod 算子 kernel 类定义
 */

#ifndef TRUNCATEMOD_H
#define TRUNCATEMOD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"
#include "truncate_mod_tiling_key.h"

namespace NsTruncateMod {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TruncateMod {
public:
    __aicore__ inline TruncateMod(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
};

template <typename T>
__aicore__ inline void TruncateMod<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    totalNum_ = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;

    // 计算当前核处理的数据范围
    int64_t blockFactor = tilingData->blockFactor;
    int64_t startIdx = blockIdx * blockFactor;
    blockLength_ = (startIdx + blockFactor > totalNum_) ? (totalNum_ - startIdx) : blockFactor;

    if (blockLength_ <= 0) {
        return;
    }

    inputGMX1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + startIdx, blockLength_);
    inputGMX2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + startIdx, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + startIdx, blockLength_);

    pipe.InitBuffer(inputQueueX1, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocalX1 = inputQueueX1.AllocTensor<T>();
    LocalTensor<T> inputLocalX2 = inputQueueX2.AllocTensor<T>();

    DataCopy(inputLocalX1, inputGMX1[progress], currentNum);
    DataCopy(inputLocalX2, inputGMX2[progress], currentNum);

    inputQueueX1.EnQue(inputLocalX1);
    inputQueueX2.EnQue(inputLocalX2);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocalX1 = inputQueueX1.DeQue<T>();
    LocalTensor<T> inputLocalX2 = inputQueueX2.DeQue<T>();
    LocalTensor<T> outputLocalY = outputQueueY.AllocTensor<T>();

    // 计算 y = x1 - trunc(x1 / x2) * x2
    // 使用 Div 计算 x1 / x2
    Div(outputLocalY, inputLocalX1, inputLocalX2, currentNum);

    // 使用 Trunc 截断到整数
    Trunc(outputLocalY, outputLocalY, currentNum);

    // 乘以 x2
    Mul(outputLocalY, outputLocalY, inputLocalX2, currentNum);

    // x1 - result
    Sub(outputLocalY, inputLocalX1, outputLocalY, currentNum);

    outputQueueY.EnQue<T>(outputLocalY);
    inputQueueX1.FreeTensor(inputLocalX1);
    inputQueueX2.FreeTensor(inputLocalX2);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocalY = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], outputLocalY, currentNum);
    outputQueueY.FreeTensor(outputLocalY);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t progress = i * ubLength_;
        int64_t currentNum = (progress + ubLength_ > blockLength_) ? (blockLength_ - progress) : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTruncateMod
#endif // TRUNCATEMOD_H
