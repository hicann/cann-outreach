// Elementwise exact-GELU approximation for contiguous ND tensors on Ascend 910B.
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                               const GeluTilingData &tiling) {
        tileLength_ = tiling.tileLength;
        constexpr uint32_t align = 32 / sizeof(DT_INPUT_X);
        const uint64_t blocks = tiling.length / align + (tiling.length % align != 0);
        const uint64_t cores = GetBlockNum();
        const uint64_t id = GetBlockIdx();
        const uint64_t base = blocks / cores;
        const uint64_t extra = blocks % cores;
        const uint64_t start = (id * base + (id < extra ? id : extra)) * align;
        const uint64_t capacity = (base + (id < extra ? 1 : 0)) * align;
        length_ = start < tiling.length ? tiling.length - start : 0;
        if (length_ > capacity) {
            length_ = capacity;
        }
        if (length_ == 0) {
            return;
        }
        inputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + start, length_);
        outputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + start, length_);
        pipe_.InitBuffer(inQueue_, 2, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueue_, 2, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(xBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(aBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(pBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(wBuf_, tileLength_ * sizeof(float));
    }
    __aicore__ inline void Process() {
        for (uint64_t offset = 0; offset < length_; offset += tileLength_) {
            const uint64_t remaining = length_ - offset;
            const uint32_t count = remaining < tileLength_ ?
                static_cast<uint32_t>(remaining) : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count) {
        auto input = inQueue_.AllocTensor<DT_INPUT_X>();
        constexpr uint32_t align = 32 / sizeof(DT_INPUT_X);
        if (count % align == 0) {
            DataCopy(input, inputGm_[offset], count);
        } else {
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> padding{
                true, 0, static_cast<uint8_t>(align - count % align), static_cast<DT_INPUT_X>(0)};
            DataCopyPad(input, inputGm_[offset], params, padding);
        }
        inQueue_.EnQue(input);
    }

    // P and Q have independent dependency chains and disjoint buffers.
    // Issue each pair before one shared barrier; preserve the rounding order
    // inside each polynomial. This requires no new buffers or math APIs.
    __aicore__ inline void HornerPair(LocalTensor<float> p,
                                     LocalTensor<float> q,
                                     LocalTensor<float> z,
                                     float cp, float cq, uint32_t count) {
        Mul(p, p, z, count);
        Mul(q, q, z, count);
        PipeBarrier<PIPE_V>();
        Adds(p, p, cp, count);
        Adds(q, q, cq, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t count) {
        constexpr uint32_t align = 32 / sizeof(DT_INPUT_X);
        const uint32_t n = ((count + align - 1) / align) * align;
        auto input = inQueue_.DeQue<DT_INPUT_X>();
        auto output = outQueue_.AllocTensor<DT_INPUT_X>();
        auto x = xBuf_.Get<float>();
        auto a = aBuf_.Get<float>();
        auto p = pBuf_.Get<float>();
        auto q = qBuf_.Get<float>();
        auto w = wBuf_.Get<float>();
        if constexpr (sizeof(DT_INPUT_X) == sizeof(half)) {
            Cast(x, input, RoundMode::CAST_NONE, n);
        } else {
            DataCopy(x, input, n);
        }
        PipeBarrier<PIPE_V>();
        Abs(a, x, n);
        PipeBarrier<PIPE_V>();
        // Clamp only the tail calculation. Preserve the original positive x.
        // This prevents overflow on finite fp32 extremes. Beyond |x|=16 the
        // negative GELU tail is below the smallest representable fp32 value.
        Mins(a, a, 16.0f, n);
        PipeBarrier<PIPE_V>();
        Muls(w, a, 0.70710678118654752440f, n);
        PipeBarrier<PIPE_V>();

        // erfc(z) ~= exp(-z*z) P(z)/Q(z), Cephes coefficients.
        // P/Q is evaluated over z in [0, 16/sqrt(2)]. This float32 extension
        // of the original interval is validated by the accompanying CPU test;
        // it is not a claim of Cephes double-precision accuracy on that domain.
        // Fold the constant initialization into the first Horner stage.
        // z*1 is exactly z for the finite, clamped z used here.
        Muls(p, w, 2.46196981473530512524e-10f, n);
        Adds(q, w, 13.2281951154744992508f, n);
        PipeBarrier<PIPE_V>();
        Adds(p, p, 5.64189564831068821977e-1f, n);
        PipeBarrier<PIPE_V>();
        HornerPair(p, q, w, 7.46321056442269912687f, 86.7072140885989742329f, n);
        HornerPair(p, q, w, 48.6371970985681366614f, 354.937778887819891062f, n);
        HornerPair(p, q, w, 196.520832956077098242f, 975.708501743205489753f, n);
        HornerPair(p, q, w, 526.445194995477358631f, 1823.90916687909736289f, n);
        HornerPair(p, q, w, 934.528527171957607540f, 2246.33760818710981792f, n);
        HornerPair(p, q, w, 1027.55188689515710272f, 1656.66309194161350182f, n);
        HornerPair(p, q, w, 557.535335369399327526f, 557.535340817727675546f, n);
        Div(p, p, q, n);
        Mul(w, a, a, n);
        PipeBarrier<PIPE_V>();
        Muls(w, w, -0.5f, n);
        PipeBarrier<PIPE_V>();
        Exp(w, w, n);
        PipeBarrier<PIPE_V>();
        Mul(p, p, w, n);
        Muls(a, a, 0.5f, n);
        PipeBarrier<PIPE_V>();
        Mul(p, p, a, n);
        Maxs(x, x, 0.0f, n);
        PipeBarrier<PIPE_V>();
        // GELU(x) = max(x, 0) - |x| * erfc(|x|/sqrt(2)) / 2.
        // Avoid 1 + erf(x/sqrt(2)) cancellation in the negative tail.
        Sub(x, x, p, n);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(DT_INPUT_X) == sizeof(half)) {
            Cast(output, x, RoundMode::CAST_RINT, n);
        } else {
            DataCopy(output, x, n);
        }
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count) {
        auto output = outQueue_.DeQue<DT_INPUT_X>();
        constexpr uint32_t align = 32 / sizeof(DT_INPUT_X);
        if (count % align == 0) {
            DataCopy(outputGm_[offset], output, count);
        } else {
            // The GM write length is the true byte count, never rounded up.
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], output, params);
        }
        outQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> xBuf_, aBuf_, pBuf_, qBuf_, wBuf_;
    GlobalTensor<DT_INPUT_X> inputGm_, outputGm_;
    uint64_t length_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
