/*!
 * \file sync_batch_norm_backward_reduce.h
 * \brief SyncBatchNormBackwardReduce 算子 kernel 类定义
 *
 * 算法（与 TBE 对齐）：
 *   dy_mean      = mean * sum_dy            (Mul)
 *   sum_dy_xmu   = sum_dy_dx_pad - dy_mean  (Sub)
 *   y            = sum_dy_xmu * invert_std  (Mul)
 * 对于 float16/bfloat16，输入先 cast 到 float32 计算，再 cast 回原 dtype。
 */

#ifndef SYNCBATCHNORMBACKWARDREDUCE_H
#define SYNCBATCHNORMBACKWARDREDUCE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_batch_norm_backward_reduce_tiling_data.h"
#include "sync_batch_norm_backward_reduce_tiling_key.h"

namespace NsSyncBatchNormBackwardReduce {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SyncBatchNormBackwardReduce {
public:
    __aicore__ inline SyncBatchNormBackwardReduce(){};

    __aicore__ inline void Init(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean, GM_ADDR invert_std,
                                GM_ADDR sum_dy_xmu, GM_ADDR y,
                                const SyncBatchNormBackwardReduceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    // 4 路输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSumDy;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSumDyDxPad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMean;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInvertStd;
    // 2 路输出队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueSumDyXmu;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> sumDyGm;
    GlobalTensor<T> sumDyDxPadGm;
    GlobalTensor<T> meanGm;
    GlobalTensor<T> invertStdGm;
    GlobalTensor<T> sumDyXmuGm;
    GlobalTensor<T> yGm;

    // float32 中间缓冲,float16/bf16 需 7 个 cast+计算缓冲
    TBuf<> sumDyFp32Buf;
    TBuf<> sumDyDxPadFp32Buf;
    TBuf<> meanFp32Buf;
    TBuf<> invertStdFp32Buf;
    TBuf<> sumDyXmuFp32Buf;
    TBuf<> yFp32Buf;

    int64_t totalNum_ = 0;//总元素数量
    int64_t ubFactor_ = 0;//每次ub数量
    int64_t myDataNum_ = 0;//当前核处理元素数量
    int64_t tileNum_ = 0;//总ub循环次数
};

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Init(
    GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean, GM_ADDR invert_std,
    GM_ADDR sum_dy_xmu, GM_ADDR y,
    const SyncBatchNormBackwardReduceTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    myDataNum_ = tilingData->blockFactor;
    ubFactor_ = tilingData->ubFactor;

    int64_t coreId = GetBlockIdx();
    int64_t offset = coreId * myDataNum_;
    if (offset + myDataNum_ > totalNum_) {
        myDataNum_ = totalNum_ - offset;
    }
    if (myDataNum_ <= 0 || ubFactor_ <= 0) {
        tileNum_ = 0;
        return;
    }

    tileNum_ = (myDataNum_ + ubFactor_ - 1) / ubFactor_;

    sumDyGm.SetGlobalBuffer((__gm__ T*)sum_dy + offset, myDataNum_);
    sumDyDxPadGm.SetGlobalBuffer((__gm__ T*)sum_dy_dx_pad + offset, myDataNum_);
    meanGm.SetGlobalBuffer((__gm__ T*)mean + offset, myDataNum_);
    invertStdGm.SetGlobalBuffer((__gm__ T*)invert_std + offset, myDataNum_);
    sumDyXmuGm.SetGlobalBuffer((__gm__ T*)sum_dy_xmu + offset, myDataNum_);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset, myDataNum_);

    pipe.InitBuffer(inQueueSumDy, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(inQueueSumDyDxPad, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(inQueueMean, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(inQueueInvertStd, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outQueueSumDyXmu, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor_ * sizeof(T));

    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(sumDyFp32Buf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(sumDyDxPadFp32Buf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(meanFp32Buf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(invertStdFp32Buf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(sumDyXmuFp32Buf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(yFp32Buf, ubFactor_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> sumDyLocal = inQueueSumDy.template AllocTensor<T>();
    LocalTensor<T> sumDyDxPadLocal = inQueueSumDyDxPad.template AllocTensor<T>();
    LocalTensor<T> meanLocal = inQueueMean.template AllocTensor<T>();
    LocalTensor<T> invertStdLocal = inQueueInvertStd.template AllocTensor<T>();

    //使用DataCopyPad
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    DataCopyPad(sumDyLocal, sumDyGm[progress * ubFactor_], copyParams, padParams);
    DataCopyPad(sumDyDxPadLocal, sumDyDxPadGm[progress * ubFactor_], copyParams, padParams);
    DataCopyPad(meanLocal, meanGm[progress * ubFactor_], copyParams, padParams);
    DataCopyPad(invertStdLocal, invertStdGm[progress * ubFactor_], copyParams, padParams);

    inQueueSumDy.EnQue(sumDyLocal);
    inQueueSumDyDxPad.EnQue(sumDyDxPadLocal);
    inQueueMean.EnQue(meanLocal);
    inQueueInvertStd.EnQue(invertStdLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Compute(int64_t currentNum)
{
    //分配local内存
    LocalTensor<T> sumDyLocal = inQueueSumDy.template DeQue<T>();
    LocalTensor<T> sumDyDxPadLocal = inQueueSumDyDxPad.template DeQue<T>();
    LocalTensor<T> meanLocal = inQueueMean.template DeQue<T>();
    LocalTensor<T> invertStdLocal = inQueueInvertStd.template DeQue<T>();
    LocalTensor<T> sumDyXmuLocal = outQueueSumDyXmu.AllocTensor<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    if constexpr (std::is_same_v<T, float>) {
        // float32 直接计算
        Mul(yLocal, meanLocal, sumDyLocal, currentNum);
        Sub(sumDyXmuLocal, sumDyDxPadLocal, yLocal, currentNum);
        Mul(yLocal, sumDyXmuLocal, invertStdLocal, currentNum);
    } else {
        // float16/bfloat16: 先 cast 到 float32
        LocalTensor<float> sumDyFp32 = sumDyFp32Buf.Get<float>();
        LocalTensor<float> sumDyDxPadFp32 = sumDyDxPadFp32Buf.Get<float>();
        LocalTensor<float> meanFp32 = meanFp32Buf.Get<float>();
        LocalTensor<float> invertStdFp32 = invertStdFp32Buf.Get<float>();
        LocalTensor<float> sumDyXmuFp32 = sumDyXmuFp32Buf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();

        Cast(sumDyFp32, sumDyLocal, RoundMode::CAST_NONE, currentNum);
        Cast(sumDyDxPadFp32, sumDyDxPadLocal, RoundMode::CAST_NONE, currentNum);
        Cast(meanFp32, meanLocal, RoundMode::CAST_NONE, currentNum);
        Cast(invertStdFp32, invertStdLocal, RoundMode::CAST_NONE, currentNum);

        // float32 计算
        Mul(yFp32, meanFp32, sumDyFp32, currentNum);
        Sub(sumDyXmuFp32, sumDyDxPadFp32, yFp32, currentNum);
        Mul(yFp32, sumDyXmuFp32, invertStdFp32, currentNum);

        // cast 回原 dtype
        Cast(sumDyXmuLocal, sumDyXmuFp32, RoundMode::CAST_RINT, currentNum);
        Cast(yLocal, yFp32, RoundMode::CAST_RINT, currentNum);
    }

    outQueueSumDyXmu.EnQue(sumDyXmuLocal);
    outQueueY.EnQue(yLocal);
    
    //释放VECIN内存以并行buffer
    inQueueSumDy.FreeTensor(sumDyLocal);
    inQueueSumDyDxPad.FreeTensor(sumDyDxPadLocal);
    inQueueMean.FreeTensor(meanLocal);
    inQueueInvertStd.FreeTensor(invertStdLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> sumDyXmuLocal = outQueueSumDyXmu.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(sumDyXmuGm[progress * ubFactor_], sumDyXmuLocal, copyParams);
    DataCopyPad(yGm[progress * ubFactor_], yLocal, copyParams);

    outQueueSumDyXmu.FreeTensor(sumDyXmuLocal);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SyncBatchNormBackwardReduce<T>::Process()
{
    if (tileNum_ == 0) {
        return;
    }

    int64_t loopCount = tileNum_ - 1;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubFactor_);
        Compute(ubFactor_);
        CopyOut(i, ubFactor_);
    }
    // 尾块
    int64_t lastNum = myDataNum_ - loopCount * ubFactor_;
    CopyIn(loopCount, lastNum);
    Compute(lastNum);
    CopyOut(loopCount, lastNum);
}

} // namespace NsSyncBatchNormBackwardReduce
#endif // SYNCBATCHNORMBACKWARDREDUCE_H
