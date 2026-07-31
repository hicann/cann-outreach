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

template <typename T, int BUFFER_MODE>
class TruncateMod {
    static constexpr int32_t BUF_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr size_t F32_SZ = sizeof(float);
    static constexpr size_t I32_SZ = sizeof(int32_t);
    static constexpr size_t F16_SZ = sizeof(half);

public:
    __aicore__ inline TruncateMod() = default;
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t p, int64_t n);
    __aicore__ inline void CopyOut(int64_t p, int64_t n);
    __aicore__ inline void Compute(int64_t n);
    __aicore__ inline void BroadcastIndex(int64_t flatIdx, int64_t* multiIdx, int ndims,
                                          const int64_t* shape);
    __aicore__ inline int64_t OffsetForBroadcast(int64_t flatIdx, const int64_t* srcShape, int srcDims,
                                                  const int64_t* outShape, int outDims);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUF_NUM> inputQueueX1;
    TQue<QuePosition::VECIN, BUF_NUM> inputQueueX2;
    TQue<QuePosition::VECOUT, BUF_NUM> outputQueueY;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    int64_t blockLen_ = 0;
    int64_t ubLen_ = 0;
    int64_t blockOffset_ = 0;
    int64_t outDims_ = 0;
    int64_t x1Dims_ = 0;
    int64_t x2Dims_ = 0;
    int64_t outShape_[8] = {0};
    int64_t x1Shape_[8] = {0};
    int64_t x2Shape_[8] = {0};
    bool x1Broadcast_ = false;
    bool x2Broadcast_ = false;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::BroadcastIndex(
    int64_t flatIdx, int64_t* multiIdx, int ndims, const int64_t* shape)
{
    for (int i = ndims - 1; i >= 0; --i) {
        multiIdx[i] = flatIdx % shape[i];
        flatIdx /= shape[i];
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline int64_t TruncateMod<T, BUFFER_MODE>::OffsetForBroadcast(
    int64_t flatIdx, const int64_t* srcShape, int srcDims,
    const int64_t* outShape, int outDims)
{
    int64_t multiIdx[8] = {0};
    BroadcastIndex(flatIdx, multiIdx, outDims, outShape);
    int64_t srcIdx = 0;
    int rankDiff = outDims - srcDims;
    for (int i = 0; i < srcDims; ++i) {
        int outDim = i + rankDiff;
        int64_t dim = (srcShape[i] == 1) ? 0 : multiIdx[outDim];
        srcIdx = srcIdx * srcShape[i] + dim;
    }
    return srcIdx;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* td)
{
    ubLen_ = td->ubFactor;
    blockOffset_ = td->blockFactor * AscendC::GetBlockIdx();
    int64_t remLen = td->totalNum - blockOffset_;
    blockLen_ = (remLen > 0) ? ((remLen > td->blockFactor) ? td->blockFactor : remLen) : 0;
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset_, blockLen_);

    outDims_ = td->outDims;
    x1Dims_ = td->x1Dims;
    x2Dims_ = td->x2Dims;
    for (int i = 0; i < 8; ++i) {
        outShape_[i] = td->outShape[i];
        x1Shape_[i] = td->x1Shape[i];
        x2Shape_[i] = td->x2Shape[i];
    }
    x1Broadcast_ = false;
    if (x1Dims_ != outDims_) {
        x1Broadcast_ = true;
    } else {
        for (int i = 0; i < x1Dims_; ++i) {
            if (x1Shape_[i] != outShape_[i]) { x1Broadcast_ = true; break; }
        }
    }
    x2Broadcast_ = false;
    if (x2Dims_ != outDims_) {
        x2Broadcast_ = true;
    } else {
        for (int i = 0; i < x2Dims_; ++i) {
            if (x2Shape_[i] != outShape_[i]) { x2Broadcast_ = true; break; }
        }
    }

    if (!x1Broadcast_) {
        inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + blockOffset_, blockLen_);
    } else {
        inputGMX1.SetGlobalBuffer((__gm__ T*)x1);
    }
    if (!x2Broadcast_) {
        inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + blockOffset_, blockLen_);
    } else {
        inputGMX2.SetGlobalBuffer((__gm__ T*)x2);
    }

    pipe.InitBuffer(inputQueueX1, BUF_NUM, ubLen_ * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUF_NUM, ubLen_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUF_NUM, ubLen_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::CopyIn(int64_t p, int64_t n)
{
    LocalTensor<T> x1l = inputQueueX1.template AllocTensor<T>();
    LocalTensor<T> x2l = inputQueueX2.template AllocTensor<T>();
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = n * sizeof(T);
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;

    if (!x1Broadcast_) {
        DataCopyPad(x1l, inputGMX1[p * ubLen_], cp, {true, 0, 0, static_cast<T>(0)});
    } else {
        int64_t baseOut = blockOffset_ + p * ubLen_;
        const __gm__ T* srcBase = inputGMX1.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            int64_t srcOff = OffsetForBroadcast(baseOut + i, x1Shape_, x1Dims_, outShape_, outDims_);
            x1l.SetValue(i, srcBase[srcOff]);
        }
    }
    if (!x2Broadcast_) {
        DataCopyPad(x2l, inputGMX2[p * ubLen_], cp, {true, 0, 0, static_cast<T>(0)});
    } else {
        int64_t baseOut = blockOffset_ + p * ubLen_;
        const __gm__ T* srcBase = inputGMX2.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            int64_t srcOff = OffsetForBroadcast(baseOut + i, x2Shape_, x2Dims_, outShape_, outDims_);
            x2l.SetValue(i, srcBase[srcOff]);
        }
    }

    inputQueueX1.EnQue(x1l);
    inputQueueX2.EnQue(x2l);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::CopyOut(int64_t p, int64_t n)
{
    LocalTensor<T> yl = outputQueueY.template DeQue<T>();
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = n * sizeof(T);
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;
    DataCopyPad(outputGMY[p * ubLen_], yl, cp);
    outputQueueY.FreeTensor(yl);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::Compute(int64_t n)
{
    LocalTensor<T> x1l = inputQueueX1.template DeQue<T>();
    LocalTensor<T> x2l = inputQueueX2.template DeQue<T>();
    LocalTensor<T> yl = outputQueueY.template AllocTensor<T>();

    size_t baseOff = BUF_NUM * 3 * ubLen_ * sizeof(T);
    baseOff = (baseOff + 31) & (~31ull);

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<int32_t> ti(TPosition::VECCALC, baseOff, n);
        AscendC::Div(yl, x1l, x2l, n);
        AscendC::Cast(ti, yl, RoundMode::CAST_TRUNC, n);
        AscendC::Cast(yl, ti, RoundMode::CAST_NONE, n);
        AscendC::Mul(yl, yl, x2l, n);
        AscendC::Sub(yl, x1l, yl, n);
    } else if constexpr (std::is_same_v<T, half>) {
        LocalTensor<float> x1f(TPosition::VECCALC, baseOff, n);
        LocalTensor<float> x2f(TPosition::VECCALC, baseOff + F32_SZ * ubLen_, n);
        LocalTensor<float> yf(TPosition::VECCALC, baseOff + 2 * F32_SZ * ubLen_, n);
        LocalTensor<int32_t> ti(TPosition::VECCALC, baseOff + 3 * F32_SZ * ubLen_, n);
        AscendC::Cast(x1f, x1l, RoundMode::CAST_NONE, n);
        AscendC::Cast(x2f, x2l, RoundMode::CAST_NONE, n);
        AscendC::Div(yf, x1f, x2f, n);
        AscendC::Cast(ti, yf, RoundMode::CAST_TRUNC, n);
        AscendC::Cast(yf, ti, RoundMode::CAST_NONE, n);
        AscendC::Mul(yf, yf, x2f, n);
        AscendC::Sub(yf, x1f, yf, n);
        AscendC::Cast(yl, yf, RoundMode::CAST_NONE, n);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> x1f(TPosition::VECCALC, baseOff, n);
        LocalTensor<float> x2f(TPosition::VECCALC, baseOff + F32_SZ * ubLen_, n);
        LocalTensor<float> yf(TPosition::VECCALC, baseOff + 2 * F32_SZ * ubLen_, n);
        LocalTensor<int32_t> ti(TPosition::VECCALC, baseOff + 3 * F32_SZ * ubLen_, n);
        AscendC::Cast(x1f, x1l, RoundMode::CAST_NONE, n);
        AscendC::Cast(x2f, x2l, RoundMode::CAST_NONE, n);
        AscendC::Div(yf, x1f, x2f, n);
        AscendC::Cast(ti, yf, RoundMode::CAST_TRUNC, n);
        AscendC::Cast(yf, ti, RoundMode::CAST_NONE, n);
        AscendC::Mul(yf, yf, x2f, n);
        AscendC::Sub(yf, x1f, yf, n);
        AscendC::Cast(yl, yf, RoundMode::CAST_ROUND, n);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        LocalTensor<float> x1f(TPosition::VECCALC, baseOff, n);
        LocalTensor<float> x2f(TPosition::VECCALC, baseOff + F32_SZ * ubLen_, n);
        LocalTensor<float> yf(TPosition::VECCALC, baseOff + 2 * F32_SZ * ubLen_, n);
        LocalTensor<int32_t> ti(TPosition::VECCALC, baseOff + 3 * F32_SZ * ubLen_, n);
        AscendC::Cast(x1f, x1l, RoundMode::CAST_NONE, n);
        AscendC::Cast(x2f, x2l, RoundMode::CAST_NONE, n);
        AscendC::Div(yf, x1f, x2f, n);
        AscendC::Cast(ti, yf, RoundMode::CAST_TRUNC, n);
        AscendC::Cast(yf, ti, RoundMode::CAST_NONE, n);
        AscendC::Mul(yf, yf, x2f, n);
        AscendC::Sub(yf, x1f, yf, n);
        AscendC::Cast(yl, yf, RoundMode::CAST_TRUNC, n);
    } else {
        size_t hOff = baseOff + 4 * F32_SZ * ubLen_ + 3 * F16_SZ * ubLen_;
        LocalTensor<float> x1f(TPosition::VECCALC, baseOff, n);
        LocalTensor<float> x2f(TPosition::VECCALC, baseOff + F32_SZ * ubLen_, n);
        LocalTensor<float> yf(TPosition::VECCALC, baseOff + 2 * F32_SZ * ubLen_, n);
        LocalTensor<int32_t> ti(TPosition::VECCALC, baseOff + 3 * F32_SZ * ubLen_, n);
        LocalTensor<half> x1h(TPosition::VECCALC, hOff, n);
        LocalTensor<half> x2h(TPosition::VECCALC, hOff + F16_SZ * ubLen_, n);
        LocalTensor<half> yh(TPosition::VECCALC, hOff + 2 * F16_SZ * ubLen_, n);
        AscendC::Cast(x1h, x1l, RoundMode::CAST_NONE, n);
        AscendC::Cast(x2h, x2l, RoundMode::CAST_NONE, n);
        AscendC::Cast(x1f, x1h, RoundMode::CAST_NONE, n);
        AscendC::Cast(x2f, x2h, RoundMode::CAST_NONE, n);
        AscendC::Div(yf, x1f, x2f, n);
        AscendC::Cast(ti, yf, RoundMode::CAST_TRUNC, n);
        AscendC::Cast(yf, ti, RoundMode::CAST_NONE, n);
        AscendC::Mul(yf, yf, x2f, n);
        AscendC::Sub(yf, x1f, yf, n);
        AscendC::Cast(yh, yf, RoundMode::CAST_NONE, n);
        AscendC::Cast(yl, yh, RoundMode::CAST_NONE, n);
    }

    outputQueueY.template EnQue<T>(yl);
    inputQueueX1.FreeTensor(x1l);
    inputQueueX2.FreeTensor(x2l);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void TruncateMod<T, BUFFER_MODE>::Process()
{
    int64_t loop = (blockLen_ + ubLen_ - 1) / ubLen_;
    for (int64_t i = 0; i < loop; i++) {
        int64_t n = (i == loop - 1) ? (blockLen_ - ubLen_ * i) : ubLen_;
        CopyIn(i, n);
        Compute(n);
        CopyOut(i, n);
    }
}

} // namespace NsTruncateMod
#endif // TRUNCATEMOD_H
