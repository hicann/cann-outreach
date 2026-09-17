/**
 * @file gcd123.cpp
 * @brief Ascend C gcd123: out = gcd(self, other)
 *
 * Inputs : self, other  — float16, ND, 4D, broadcast-compatible
 * Output : out          — float16, ND, shape = broadcast(self, other)
 *
 * Demo case (broadcast materialized on host):
 *   self  [1, 4, 1, 128]
 *   other [1, 4, 16, 1]
 *   out   [1, 4, 16, 128]  -> TOTAL_LENGTH = 8192
 *
 * Algorithm: cast half->float, abs, Euclidean GCD (16 vectorized iters), cast back.
 */
#include "kernel_operator.h"
#include "gcd123_tiling.h"

constexpr int32_t TOTAL_LENGTH = 1 * 4 * 16 * 128; // 8192, broadcasted out shape
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM; // 1024
constexpr int32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;     // 1024/256 = 4

class KernelGcd123 {
public:
    __aicore__ inline KernelGcd123() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out)
    {
        selfGm.SetGlobalBuffer((__gm__ half *)self + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        otherGm.SetGlobalBuffer((__gm__ half *)other + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        outGm.SetGlobalBuffer((__gm__ half *)out + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);

        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueOther, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(tmpBufA, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBufB, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBufQ, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBufR, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(cmpBuf, TILE_LENGTH * sizeof(uint8_t));
        pipe.InitBuffer(zeroBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpI32Buf, TILE_LENGTH * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < TILE_NUM; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> selfLocal = inQueueSelf.AllocTensor<half>();
        AscendC::LocalTensor<half> otherLocal = inQueueOther.AllocTensor<half>();
        AscendC::DataCopy(selfLocal, selfGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(otherLocal, otherGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueSelf.EnQue(selfLocal);
        inQueueOther.EnQue(otherLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> selfLocal = inQueueSelf.DeQue<half>();
        AscendC::LocalTensor<half> otherLocal = inQueueOther.DeQue<half>();
        AscendC::LocalTensor<half> outLocal = outQueueOut.AllocTensor<half>();

        AscendC::LocalTensor<float> a = tmpBufA.Get<float>();
        AscendC::LocalTensor<float> b = tmpBufB.Get<float>();
        AscendC::LocalTensor<float> tmpQ = tmpBufQ.Get<float>();
        AscendC::LocalTensor<float> tmpR = tmpBufR.Get<float>();
        AscendC::LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> zero = zeroBuf.Get<float>();
        AscendC::LocalTensor<int32_t> tmpI32 = tmpI32Buf.Get<int32_t>();

        AscendC::Duplicate(zero, static_cast<float>(0.0f), TILE_LENGTH);
        AscendC::Cast(a, selfLocal, AscendC::RoundMode::CAST_NONE, TILE_LENGTH);
        AscendC::Cast(b, otherLocal, AscendC::RoundMode::CAST_NONE, TILE_LENGTH);
        AscendC::Abs(a, a, TILE_LENGTH);
        AscendC::Abs(b, b, TILE_LENGTH);

        for (uint32_t iter = 0; iter < EUCLID_ITERS; iter++) {
            AscendC::Compare(cmp, b, zero, AscendC::CMPMODE::GT, TILE_LENGTH);
            AscendC::Div(tmpQ, a, b, TILE_LENGTH);
            AscendC::Cast(tmpI32, tmpQ, AscendC::RoundMode::CAST_TRUNC, TILE_LENGTH);
            AscendC::Cast(tmpQ, tmpI32, AscendC::RoundMode::CAST_NONE, TILE_LENGTH);
            AscendC::Mul(tmpR, tmpQ, b, TILE_LENGTH);
            AscendC::Sub(tmpR, a, tmpR, TILE_LENGTH);
            AscendC::Select(a, cmp, b, a, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_LENGTH);
            AscendC::Select(b, cmp, tmpR, b, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_LENGTH);
        }

        AscendC::Cast(outLocal, a, AscendC::RoundMode::CAST_NONE, TILE_LENGTH);
        outQueueOut.EnQue<half>(outLocal);
        inQueueSelf.FreeTensor(selfLocal);
        inQueueOther.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> outLocal = outQueueOut.DeQue<half>();
        AscendC::DataCopy(outGm[progress * TILE_LENGTH], outLocal, TILE_LENGTH);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueSelf;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueOther;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufR;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpI32Buf;
    AscendC::GlobalTensor<half> selfGm;
    AscendC::GlobalTensor<half> otherGm;
    AscendC::GlobalTensor<half> outGm;
};

extern "C" __global__ __aicore__ void gcd123(GM_ADDR self, GM_ADDR other, GM_ADDR out)
{
    KernelGcd123 op;
    op.Init(self, other, out);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gcd123_do(uint32_t blockDim, void *stream, uint8_t *self, uint8_t *other, uint8_t *out)
{
    gcd123<<<blockDim, nullptr, stream>>>(self, other, out);
}
#endif
