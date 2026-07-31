/*!
 * \file truncate_mod.h
 * \brief TruncateMod 算子 kernel 类定义
 *
 * y = TruncateMod(x1, x2) = x1 - trunc(x1 / x2) * x2
 * trunc 为向零取整，余数符号跟随被除数 x1（C fmod / TF truncate_mod 语义）。
 * 所有运算在 float32 中完成后转回目标类型：float32 直接算，float16 提升到 fp32。
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
    // float32 计算缓存（half 路径提升到 fp32 时使用）。
    TBuf<QuePosition::VECCALC> calcBufF1;
    TBuf<QuePosition::VECCALC> calcBufF2;
    TBuf<QuePosition::VECCALC> calcBufF3;
    // 向零取整用的 int32 缓存。
    TBuf<QuePosition::VECCALC> calcBufI;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核处理的元素数
    int64_t ubLength_ = 0;     // 每次 UB 循环处理的元素数
    int64_t blockFactor_ = 0;  // 每个核处理的元素数（用于 GM 偏移）
};

template <typename T>
__aicore__ inline void TruncateMod<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                            const TruncateModTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t blockIdx = GetBlockIdx();
    int64_t blockNum = GetBlockNum();
    // 前 blockNum-1 个核各处理 blockFactor_ 个元素，最后一个核处理剩余部分。
    if (blockIdx < blockNum - 1) {
        blockLength_ = blockFactor_;
    } else {
        blockLength_ = totalNum - blockFactor_ * (blockNum - 1);
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    int64_t gmOffset = blockIdx * blockFactor_;
    inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + gmOffset, blockLength_);
    inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    pipe.InitBuffer(inputQueueX1, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(calcBufF1, ubLength_ * sizeof(float));
    pipe.InitBuffer(calcBufF2, ubLength_ * sizeof(float));
    pipe.InitBuffer(calcBufF3, ubLength_ * sizeof(float));
    pipe.InitBuffer(calcBufI, ubLength_ * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> x1Local = inputQueueX1.AllocTensor<T>();
    LocalTensor<T> x2Local = inputQueueX2.AllocTensor<T>();
    // 用 DataCopyPad 精确搬运 currentNum 个元素，兼容非 32B 对齐的尾块。
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(x1Local, inputGMX1[progress * ubLength_], copyParams, padParams);
    DataCopyPad(x2Local, inputGMX2[progress * ubLength_], copyParams, padParams);
    inputQueueX1.EnQue(x1Local);
    inputQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> x1Local = inputQueueX1.DeQue<T>();
    LocalTensor<T> x2Local = inputQueueX2.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    int32_t n = static_cast<int32_t>(currentNum);
    LocalTensor<int32_t> iTmp = calcBufI.Get<int32_t>();

    if constexpr (IsSameType<T, float>::value) {
        LocalTensor<float> fTmp = calcBufF1.Get<float>();
        // quot = x1 / x2 （暂存到 yLocal）
        Div(yLocal, x1Local, x2Local, n);
        // trunc(quot)：float -> int32(向零取整) -> float
        Cast(iTmp, yLocal, RoundMode::CAST_TRUNC, n);
        Cast(fTmp, iTmp, RoundMode::CAST_NONE, n);
        // fTmp = trunc(quot) * x2
        Mul(fTmp, fTmp, x2Local, n);
        // y = x1 - trunc(quot) * x2
        Sub(yLocal, x1Local, fTmp, n);
    } else {
        // float16 / bfloat16：提升到 float32 计算以满足精度。
        LocalTensor<float> x1f = calcBufF1.Get<float>();
        LocalTensor<float> x2f = calcBufF2.Get<float>();
        LocalTensor<float> quot = calcBufF3.Get<float>();
        Cast(x1f, x1Local, RoundMode::CAST_NONE, n);
        Cast(x2f, x2Local, RoundMode::CAST_NONE, n);
        Div(quot, x1f, x2f, n);
        Cast(iTmp, quot, RoundMode::CAST_TRUNC, n);
        Cast(quot, iTmp, RoundMode::CAST_NONE, n);
        Mul(quot, quot, x2f, n);
        Sub(x1f, x1f, quot, n);
        Cast(yLocal, x1f, RoundMode::CAST_RINT, n);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX1.FreeTensor(x1Local);
    inputQueueX2.FreeTensor(x2Local);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Process()
{
    if (ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    int64_t remain = blockLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (remain < ubLength_) ? remain : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
        remain -= currentNum;
    }
}

} // namespace NsTruncateMod
#endif // TRUNCATEMOD_H
