/*!
 * \file truncate_mod.h
 * \brief TruncateMod: y = x1 - trunc(x1/x2) * x2  (fp16 / bf16 / fp32)
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
constexpr int64_t DATA_BLOCK = 64;

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
    TQue<QuePosition::VECIN,  BUFFER_NUM> inputQueueX, inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> x1F32Buf, x2F32Buf;
    TBuf<TPosition::VECCALC> floorBuf, ceilBuf, negBuf, zeroBuf, tqBuf, prodBuf, remBuf;
    TBuf<TPosition::VECCALC> scratchBuf, maskBuf;
    GlobalTensor<T> inputGMX, inputGMY, outputGM;
    int64_t blockLength_ = 0, ubLength_ = 0, tileNum_ = 0, lastLen_ = 0;
    int64_t blockOff_ = 0, totalNum_ = 0;
    int64_t x1Total_ = 0, x2Total_ = 0;
    int64_t x1LastDim_ = 0, x2LastDim_ = 0, outLastDim_ = 0;
};

template <typename T>
__aicore__ inline void TruncateMod<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* td)
{
    int64_t total  = td->totalNum, bf = td->blockFactor, uf = td->ubFactor;
    int64_t bidx   = GetBlockIdx();
    int64_t offset = bidx * bf, remain = total - offset;
    if (remain < bf) bf = remain;

    blockLength_ = bf;
    ubLength_    = (uf < bf) ? uf : bf;
    if (ubLength_ < DATA_BLOCK) ubLength_ = DATA_BLOCK;
    constexpr int64_t DB = std::is_same_v<T, bfloat16_t> ? 16 : 64;
    ubLength_    = ((ubLength_ + DB - 1) / DB) * DB;
    tileNum_     = bf / ubLength_;
    lastLen_     = bf - tileNum_ * ubLength_;
    blockOff_    = offset; totalNum_ = total;
    x1Total_ = td->x1Total; x2Total_ = td->x2Total;
    x1LastDim_ = td->x1LastDim; x2LastDim_ = td->x2LastDim; outLastDim_ = td->outLastDim;

    inputGMX.SetGlobalBuffer((__gm__ T*)x1 + offset, bf);
    inputGMY.SetGlobalBuffer((__gm__ T*)x2 + offset, bf);
    outputGM.SetGlobalBuffer((__gm__ T*)y  + offset, bf);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(x1F32Buf,  ubLength_ * sizeof(float));
    pipe.InitBuffer(x2F32Buf,  ubLength_ * sizeof(float));
    pipe.InitBuffer(floorBuf,  ubLength_ * sizeof(float));
    pipe.InitBuffer(ceilBuf,   ubLength_ * sizeof(float));
    pipe.InitBuffer(negBuf,    ubLength_ * sizeof(float));
    pipe.InitBuffer(zeroBuf,   ubLength_ * sizeof(float));
    pipe.InitBuffer(tqBuf,     ubLength_ * sizeof(float));
    pipe.InitBuffer(prodBuf,   ubLength_ * sizeof(float));
    pipe.InitBuffer(remBuf,    ubLength_ * sizeof(float));
    pipe.InitBuffer(scratchBuf, ubLength_ * sizeof(uint8_t));
    pipe.InitBuffer(maskBuf,    ubLength_ * sizeof(uint8_t));
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Process()
{
    if (tileNum_ == 0 && lastLen_ == 0) return;
    if (tileNum_ > 0) {
        CopyIn(0, ubLength_);
        for (int64_t i = 0; i < tileNum_ - 1; i++) { Compute(ubLength_); CopyOut(i, ubLength_); CopyIn(i + 1, ubLength_); }
        Compute(ubLength_); CopyOut(tileNum_ - 1, ubLength_);
    }
    if (lastLen_ > 0) { CopyIn(tileNum_, lastLen_); Compute(lastLen_); CopyOut(tileNum_, lastLen_); }
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    int64_t off = progress * ubLength_;
    LocalTensor<T> x1L = inputQueueX.AllocTensor<T>();
    LocalTensor<T> x2L = inputQueueY.AllocTensor<T>();
    DataCopyParams cp; cp.blockCount = 1; cp.blockLen = currentNum * sizeof(T); cp.srcStride = 0; cp.dstStride = 0;
    DataCopyPad(x1L, inputGMX[off], cp, {false, 0, 0, 0});
    DataCopyPad(x2L, inputGMY[off], cp, {false, 0, 0, 0});
    inputQueueX.EnQue(x1L); inputQueueY.EnQue(x2L);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Compute(int64_t n)
{
    LocalTensor<T> x1L = inputQueueX.DeQue<T>(), x2L = inputQueueY.DeQue<T>();
    LocalTensor<T> yL  = outputQueue.AllocTensor<T>();
    constexpr int64_t DB = std::is_same_v<T, bfloat16_t> ? 16 : 64;
    int64_t an = ((n + DB - 1) / DB) * DB;

    LocalTensor<float> x1F, x2F;
    if constexpr (std::is_same_v<T, float>) {
        x1F = x1L.template ReinterpretCast<float>();
        x2F = x2L.template ReinterpretCast<float>();
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        x1F = x1F32Buf.Get<float>(); x2F = x2F32Buf.Get<float>();
        Duplicate(x1F, 0.0f, an);
        Duplicate(x2F, 1.0f, an);
        Cast(x1F, x1L, RoundMode::CAST_NONE, n);
        Cast(x2F, x2L, RoundMode::CAST_NONE, n);
    } else {
        x1F = x1F32Buf.Get<float>(); x2F = x2F32Buf.Get<float>();
        Duplicate(x1F, 0.0f, an);
        Duplicate(x2F, 1.0f, an);
        Cast(x1F, x1L, RoundMode::CAST_NONE, n);
        Cast(x2F, x2L, RoundMode::CAST_NONE, n);
    }
    LocalTensor<float> fQ = floorBuf.Get<float>(), cQ = ceilBuf.Get<float>();
    LocalTensor<float> nQ = negBuf.Get<float>(),   zF = zeroBuf.Get<float>();
    LocalTensor<float> tQ = tqBuf.Get<float>(),    pQ = prodBuf.Get<float>();
    LocalTensor<float> rF = remBuf.Get<float>();
    LocalTensor<uint8_t> sc = scratchBuf.Get<uint8_t>(), mk = maskBuf.Get<uint8_t>();
    Div(rF, x1F, x2F, an);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<int32_t> qI = ceilBuf.Get<int32_t>();
        Cast(qI, rF, RoundMode::CAST_TRUNC, an);
        Cast(tQ, qI, RoundMode::CAST_RINT, an);
    } else {
        Floor(fQ, rF, sc, an); Muls(nQ, rF, -1.0f, an);
        Floor(cQ, nQ, sc, an); Muls(cQ, cQ, -1.0f, an);
        Duplicate(zF, 0.0f, an);
        Compare(mk, rF, zF, CMPMODE::GE, an);
        Select(tQ, mk, fQ, cQ, SELMODE::VSEL_TENSOR_TENSOR_MODE, an);
    }
    Mul(pQ, tQ, x2F, an); Sub(rF, x1F, pQ, an);

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> zT = zeroBuf.Get<float>(); Add(yL, rF, zT, n);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        Cast(yL, rF, RoundMode::CAST_RINT, n);
    } else {
        Cast(yL, rF, RoundMode::CAST_TRUNC, n);
    }
    outputQueue.EnQue<T>(yL);
    inputQueueX.FreeTensor(x1L); inputQueueY.FreeTensor(x2L);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    int64_t off = progress * ubLength_;
    LocalTensor<T> yL = outputQueue.DeQue<T>();
    DataCopyParams cp; cp.blockCount = 1; cp.blockLen = currentNum * sizeof(T); cp.srcStride = 0; cp.dstStride = 0;
    DataCopyPad(outputGM[off], yL, cp);
    outputQueue.FreeTensor(yL);
}

} // namespace NsTruncateMod
#endif
