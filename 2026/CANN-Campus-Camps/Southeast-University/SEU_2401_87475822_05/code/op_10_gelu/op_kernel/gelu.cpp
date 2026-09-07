// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        // Split on float-vector boundaries; the last core alone owns the tail.
        constexpr uint32_t ALIGN = 64;
        uint32_t blocks = (length + ALIGN - 1) / ALIGN;
        uint32_t cores = GetBlockNum();
        uint32_t core = GetBlockIdx();
        uint32_t base = blocks / cores;
        uint32_t extra = blocks % cores;
        uint32_t start = (core * base + (core < extra ? core : extra)) * ALIGN;
        uint32_t assigned = (base + (core < extra ? 1 : 0)) * ALIGN;
        count_ = start < length ? length - start : 0;
        if (count_ > assigned) count_ = assigned;
        if (count_ == 0) return;
        xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + start, count_);
        yGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + start, count_);
        uint32_t capacity = count_ < TILE ? (count_ + ALIGN - 1) / ALIGN * ALIGN : TILE;
        uint32_t buffers = count_ > TILE ? 2 : 1;
        pipe_.InitBuffer(inQueue_, buffers, capacity * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueue_, buffers, capacity * sizeof(DT_INPUT_X));
        if constexpr (sizeof(DT_INPUT_X) != sizeof(float)) {
            pipe_.InitBuffer(xBuf_, capacity * sizeof(float));
        }
        pipe_.InitBuffer(aBuf_, capacity * sizeof(float));
        pipe_.InitBuffer(zBuf_, capacity * sizeof(float));
        pipe_.InitBuffer(pBuf_, capacity * sizeof(float));
    }
    __aicore__ inline void Process() {
        if (count_ == 0) return;
        CopyIn(0, count_ < TILE ? count_ : TILE);
        for (uint32_t offset = 0; offset < count_; offset += TILE) {
            uint32_t n = count_ - offset;
            if (n > TILE) n = TILE;
            // Prefetch the next tile while the current tile is being consumed.
            uint32_t next = offset + TILE;
            if (next < count_) {
                uint32_t nextCount = count_ - next;
                CopyIn(next, nextCount < TILE ? nextCount : TILE);
            }
            Compute(n);
            CopyOut(offset, n);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t n) {
        LocalTensor<DT_INPUT_X> input = inQueue_.AllocTensor<DT_INPUT_X>();
        constexpr uint32_t BLOCK = 32 / sizeof(DT_INPUT_X);
        if (n % BLOCK == 0) {
            DataCopy(input, xGm_[offset], n);
        } else {
            DataCopyExtParams copy = {1, n * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> pad = {false, 0, 0, 0};
            DataCopyPad(input, xGm_[offset], copy, pad);
        }
        inQueue_.EnQue(input);
    }
    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<DT_INPUT_X> input = inQueue_.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> output = outQueue_.AllocTensor<DT_INPUT_X>();
        LocalTensor<float> x;
        LocalTensor<float> result;
        if constexpr (sizeof(DT_INPUT_X) == sizeof(float)) {
            // Use queue storage directly: no float32 input/output copies.
            x = input.template ReinterpretCast<float>();
            result = output.template ReinterpretCast<float>();
        } else {
            x = xBuf_.Get<float>();
            result = x;
            Cast(x, input, RoundMode::CAST_NONE, n);
        }
        LocalTensor<float> a = aBuf_.Get<float>();
        LocalTensor<float> z = zBuf_.Get<float>();
        LocalTensor<float> p = pBuf_.Get<float>();
        // GELU(x) = max(x, 0) - a * Phi(-a), a = abs(x).
        // Approximate log(Phi(a)/Phi(-a)) by an odd minimax polynomial.
        // Clamp only the small correction, keeping large positive x intact.
        Abs(a, x, n);
        Mins(a, a, 6.0f, n);
        Mul(z, a, a, n);
        if constexpr (sizeof(DT_INPUT_X) == sizeof(float)) {
            // Degree 9: float32 absolute error below 3.3e-6 in CPU checks.
            Muls(p, z, -2.225263187198141e-6f, n);
            Adds(p, p, 6.093439338766565e-5f, n);
            Mul(p, p, z, n);
            Adds(p, p, 2.5095241553991783e-4f, n);
            Mul(p, p, z, n);
            Adds(p, p, -0.07293986152942938f, n);
            Mul(p, p, z, n);
            Adds(p, p, -1.5956551532991676f, n);
        } else {
            // Degree 5, with float16 rounding-interval constraints.
            Muls(p, z, 7.869662188642361e-4f, n);
            Adds(p, p, -0.07463140817340211f, n);
            Mul(p, p, z, n);
            Adds(p, p, -1.594344651329272f, n);
        }
        Mul(p, p, a, n);
        Exp(p, p, n);
        Adds(z, p, 1.0f, n);
        Mul(p, p, a, n);
        Div(p, p, z, n);
        Maxs(a, x, 0.0f, n);
        Sub(result, a, p, n);
        if constexpr (sizeof(DT_INPUT_X) != sizeof(float)) {
            Cast(output, result, RoundMode::CAST_NONE, n);
        }
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t n) {
        LocalTensor<DT_INPUT_X> output = outQueue_.DeQue<DT_INPUT_X>();
        constexpr uint32_t BLOCK = 32 / sizeof(DT_INPUT_X);
        if (n % BLOCK == 0) {
            DataCopy(yGm_[offset], output, n);
        } else {
            DataCopyExtParams copy = {1, n * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
            DataCopyPad(yGm_[offset], output, copy);
        }
        outQueue_.FreeTensor(output);
    }
    static constexpr uint32_t TILE = 4096;
    uint32_t count_ = 0;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<QuePosition::VECCALC> xBuf_, aBuf_, zBuf_, pBuf_;
    GlobalTensor<DT_INPUT_X> xGm_, yGm_;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}
