/*!
 * \file sync_batch_norm_backward_reduce.h
 * \brief SyncBatchNormBackwardReduce 算子 kernel 类定义
 */

#ifndef SYNCBATCHNORMBACKWARDREDUCE_H
#define SYNCBATCHNORMBACKWARDREDUCE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_batch_norm_backward_reduce_tiling_data.h"
#include "sync_batch_norm_backward_reduce_tiling_key.h"
#include <cstdint>
#include <type_traits>

namespace NsSyncBatchNormBackwardReduce {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline bfloat16_t RoundToBfloat16(float value)
{
    union FloatBits {
        __aicore__ FloatBits() {}
        float fp;
        uint32_t bits;
    } in;
    union BfloatBits {
        __aicore__ BfloatBits() {}
        uint16_t bits;
        bfloat16_t value;
    } out;

    in.fp = value;
    uint32_t absBits = in.bits & 0x7fffffffU;
    if (absBits > 0x7f800000U) {
        out.bits = static_cast<uint16_t>(in.bits >> 16);
        if ((out.bits & 0x007fU) == 0) {
            out.bits = static_cast<uint16_t>(out.bits | 0x0001U);
        }
        return out.value;
    }

    // Match vector CAST_RINT: round FP32 to BF16 with round-to-nearest-even.
    uint32_t lsb = (in.bits >> 16) & 1U;
    out.bits = static_cast<uint16_t>((in.bits + 0x7fffU + lsb) >> 16);
    return out.value;
}

template <typename T>
class SyncBatchNormBackwardReduce {
public:
    __aicore__ inline SyncBatchNormBackwardReduce(){};

    __aicore__ inline void Init(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean, GM_ADDR invert_std, GM_ADDR sum_dy_xmu, GM_ADDR y, const SyncBatchNormBackwardReduceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ReadValue(const LocalTensor<T>& tensor, int64_t index) const;
    __aicore__ inline T CastFromFloat(float value) const;
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> sumDyQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> sumDyDxPadQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> invertStdQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sumDyXmuQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> calcBuffer;

    GlobalTensor<T> sumDyGM;
    GlobalTensor<T> sumDyDxPadGM;
    GlobalTensor<T> meanGM;
    GlobalTensor<T> invertStdGM;
    GlobalTensor<T> sumDyXmuGM;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t blockOffset_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Init(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean, GM_ADDR invert_std, GM_ADDR sum_dy_xmu, GM_ADDR y, const SyncBatchNormBackwardReduceTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockOffset_ = tilingData->blockFactor * GetBlockIdx();
    int64_t remain = totalNum_ - blockOffset_;
    blockLength_ = remain > tilingData->blockFactor ? tilingData->blockFactor : remain;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    sumDyGM.SetGlobalBuffer((__gm__ T*)sum_dy, totalNum_);
    sumDyDxPadGM.SetGlobalBuffer((__gm__ T*)sum_dy_dx_pad, totalNum_);
    meanGM.SetGlobalBuffer((__gm__ T*)mean, totalNum_);
    invertStdGM.SetGlobalBuffer((__gm__ T*)invert_std, totalNum_);
    sumDyXmuGM.SetGlobalBuffer((__gm__ T*)sum_dy_xmu, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > 0) {
        pipe.InitBuffer(sumDyQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(sumDyDxPadQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(meanQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(invertStdQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(sumDyXmuQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(calcBuffer, ubLength_ * sizeof(float) * 6);
        }
    }
}

template <typename T>
__aicore__ inline float SyncBatchNormBackwardReduce<T>::ReadValue(const LocalTensor<T>& tensor, int64_t index) const
{
    T value = tensor.GetValue(index);
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T SyncBatchNormBackwardReduce<T>::CastFromFloat(float value) const
{
    if constexpr (std::is_same<T, float>::value) {
        return value;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return RoundToBfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> sumDyLocal = sumDyQueue.template AllocTensor<T>();
    LocalTensor<T> sumDyDxPadLocal = sumDyDxPadQueue.template AllocTensor<T>();
    LocalTensor<T> meanLocal = meanQueue.template AllocTensor<T>();
    LocalTensor<T> invertStdLocal = invertStdQueue.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sumDyLocal, sumDyGM[offset], copyParams, padParams);
    DataCopyPad(sumDyDxPadLocal, sumDyDxPadGM[offset], copyParams, padParams);
    DataCopyPad(meanLocal, meanGM[offset], copyParams, padParams);
    DataCopyPad(invertStdLocal, invertStdGM[offset], copyParams, padParams);
    sumDyQueue.EnQue(sumDyLocal);
    sumDyDxPadQueue.EnQue(sumDyDxPadLocal);
    meanQueue.EnQue(meanLocal);
    invertStdQueue.EnQue(invertStdLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> sumDyLocal = sumDyQueue.template DeQue<T>();
    LocalTensor<T> sumDyDxPadLocal = sumDyDxPadQueue.template DeQue<T>();
    LocalTensor<T> meanLocal = meanQueue.template DeQue<T>();
    LocalTensor<T> invertStdLocal = invertStdQueue.template DeQue<T>();
    LocalTensor<T> sumDyXmuLocal = sumDyXmuQueue.template AllocTensor<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        Mul(yLocal, meanLocal, sumDyLocal, currentNum);
        Sub(sumDyXmuLocal, sumDyDxPadLocal, yLocal, currentNum);
        Mul(yLocal, sumDyXmuLocal, invertStdLocal, currentNum);
    } else {
        LocalTensor<float> calc = calcBuffer.Get<float>();
        LocalTensor<float> sumDyFp32 = calc;
        LocalTensor<float> sumDyDxPadFp32 = calc[ubLength_];
        LocalTensor<float> meanFp32 = calc[ubLength_ * 2];
        LocalTensor<float> invertStdFp32 = calc[ubLength_ * 3];
        LocalTensor<float> sumDyXmuFp32 = calc[ubLength_ * 4];
        LocalTensor<float> yFp32 = calc[ubLength_ * 5];
        Cast(sumDyFp32, sumDyLocal, RoundMode::CAST_NONE, currentNum);
        Cast(sumDyDxPadFp32, sumDyDxPadLocal, RoundMode::CAST_NONE, currentNum);
        Cast(meanFp32, meanLocal, RoundMode::CAST_NONE, currentNum);
        Cast(invertStdFp32, invertStdLocal, RoundMode::CAST_NONE, currentNum);
        Mul(yFp32, meanFp32, sumDyFp32, currentNum);
        Sub(sumDyXmuFp32, sumDyDxPadFp32, yFp32, currentNum);
        Mul(yFp32, sumDyXmuFp32, invertStdFp32, currentNum);
        Cast(sumDyXmuLocal, sumDyXmuFp32, RoundMode::CAST_RINT, currentNum);
        Cast(yLocal, yFp32, RoundMode::CAST_RINT, currentNum);
    }
    sumDyXmuQueue.EnQue(sumDyXmuLocal);
    outputQueue.EnQue(yLocal);
    sumDyQueue.FreeTensor(sumDyLocal);
    sumDyDxPadQueue.FreeTensor(sumDyDxPadLocal);
    meanQueue.FreeTensor(meanLocal);
    invertStdQueue.FreeTensor(invertStdLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> sumDyXmuLocal = sumDyXmuQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(sumDyXmuGM[offset], sumDyXmuLocal, copyParams);
    DataCopyPad(outputGMY[offset], yLocal, copyParams);
    sumDyXmuQueue.FreeTensor(sumDyXmuLocal);
    outputQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        int64_t offset = blockOffset_ + i * ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsSyncBatchNormBackwardReduce
#endif // SYNCBATCHNORMBACKWARDREDUCE_H
