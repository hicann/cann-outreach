/**
 * @file gcd123.cpp
 *
 * AscendC kernel for the Gcd123 operator.
 *
 * Computes, element-wise over a 4D ND broadcast:
 *   out[i] = gcd(round|self[i]|, round|other[i]|)
 *
 * self and other are float16. GCD is an integer operation, so each half
 * value is rounded to the nearest integer (ties away from zero), its
 * absolute value is taken, and the Euclidean algorithm is applied; the
 * result is written back as float16. 0 is handled naturally:
 *   gcd(0, a) == gcd(a, 0) == |a|,  gcd(0, 0) == 0.
 *
 * Parallelism:
 *   - The flattened output index range [0, totalLength) is split across the
 *     AI cores. Every core's range starts on a 64-byte boundary, which is
 *     required because GM scalar writes (GlobalTensor::SetValue) dirty
 *     64-byte DataCache lines; two cores writing into the same line can
 *     clobber each other's dirty copies.
 *   - For each linear output index the kernel maps it to 4D indices
 *     (i4, i3, i2, i1) in the output shape, then to the corresponding
 *     self/other linear indices using the broadcast relation (an input
 *     dimension of 1 collapses to index 0).
 *
 * Correctness note: element-wise scalar GM access is correct for all
 * broadcast shapes and is the simplest portable implementation. For very
 * large tensors it can be reworked into a vectorized UB pipeline without
 * changing the tiling contract.
 */
#include "kernel_operator.h"

#include "../../op_host/gcd123_tiling.h"

namespace {

// 64-byte DataCache line expressed in float16 elements.
constexpr uint64_t HALF_CACHE_LINE_ELEMS = 32;

// Euclidean GCD.
__aicore__ inline uint32_t EuclidGcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// float16 -> |round(value)| as uint32_t. Robust to NaN/Inf inputs.
__aicore__ inline uint32_t HalfToAbsInt(half value)
{
    const float fv = static_cast<float>(value);
    if (fv != fv) {
        return 0; // NaN -> 0
    }
    const float mag = (fv < 0.0f) ? -fv : fv;
    if (mag > 65504.0f) {
        return 65504; // +/-Inf -> clamp to the float16 max
    }
    // Round half away from zero; float16 max is 65504, well inside int32.
    const float rounded = (fv >= 0.0f) ? (fv + 0.5f) : (fv - 0.5f);
    const int32_t iv = static_cast<int32_t>(rounded);
    return (iv < 0) ? static_cast<uint32_t>(-iv) : static_cast<uint32_t>(iv);
}

// Row-major element strides for a 4D shape.
__aicore__ inline void FillStrides(const uint32_t n[4], uint64_t s[4])
{
    s[3] = 1;
    s[2] = static_cast<uint64_t>(n[3]);
    s[1] = s[2] * static_cast<uint64_t>(n[2]);
    s[0] = s[1] * static_cast<uint64_t>(n[1]);
}

struct KernelGcd123 {
    uint64_t outStrides[4];   // row-major strides of the output
    uint64_t selfStrides[4];  // row-major strides of self
    uint64_t otherStrides[4]; // row-major strides of other
    uint32_t selfN[4];
    uint32_t otherN[4];
    uint64_t blockStart;
    uint64_t blockEnd;

    AscendC::GlobalTensor<half> selfGm;
    AscendC::GlobalTensor<half> otherGm;
    AscendC::GlobalTensor<half> outGm;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                const optiling::Gcd123TilingData &tiling)
    {
        const uint64_t totalLength = tiling.totalLength;

        selfN[0] = tiling.selfN4;
        selfN[1] = tiling.selfN3;
        selfN[2] = tiling.selfN2;
        selfN[3] = tiling.selfN1;
        otherN[0] = tiling.otherN4;
        otherN[1] = tiling.otherN3;
        otherN[2] = tiling.otherN2;
        otherN[3] = tiling.otherN1;
        const uint32_t outN[4] = {tiling.outN4, tiling.outN3, tiling.outN2, tiling.outN1};

        FillStrides(selfN, selfStrides);
        FillStrides(otherN, otherStrides);
        FillStrides(outN, outStrides);

        // Even split with a remainder, each core range aligned to a
        // 64-byte DataCache line (see file header).
        const uint64_t blockNum = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        uint64_t blockLength = (totalLength + blockNum - 1) / blockNum;
        blockLength = (blockLength + HALF_CACHE_LINE_ELEMS - 1) / HALF_CACHE_LINE_ELEMS *
                      HALF_CACHE_LINE_ELEMS;
        this->blockStart = static_cast<uint64_t>(blockIdx) * blockLength;
        this->blockEnd = this->blockStart + blockLength;
        if (this->blockEnd > totalLength) {
            this->blockEnd = totalLength;
        }
        if (this->blockStart > totalLength) {
            this->blockStart = totalLength;
        }

        selfGm.SetGlobalBuffer((__gm__ half *)self);
        otherGm.SetGlobalBuffer((__gm__ half *)other);
        outGm.SetGlobalBuffer((__gm__ half *)out);
    }

    __aicore__ inline void Process()
    {
        const uint64_t s0 = outStrides[0];
        const uint64_t s1 = outStrides[1];
        const uint64_t s2 = outStrides[2];
        for (uint64_t lin = this->blockStart; lin < this->blockEnd; ++lin) {
            // Split the linear output index into 4D indices (i4, i3, i2, i1).
            uint64_t i4 = lin / s0;
            uint64_t r = lin % s0;
            uint64_t i3 = r / s1;
            r = r % s1;
            uint64_t i2 = r / s2;
            uint64_t i1 = r % s2;

            // Broadcast mapping: an input dim of 1 collapses to index 0.
            uint64_t selfIdx = 0;
            uint64_t otherIdx = 0;
            if (this->selfN[0] > 1) {
                selfIdx += i4 * this->selfStrides[0];
            }
            if (this->selfN[1] > 1) {
                selfIdx += i3 * this->selfStrides[1];
            }
            if (this->selfN[2] > 1) {
                selfIdx += i2 * this->selfStrides[2];
            }
            if (this->selfN[3] > 1) {
                selfIdx += i1 * this->selfStrides[3];
            }
            if (this->otherN[0] > 1) {
                otherIdx += i4 * this->otherStrides[0];
            }
            if (this->otherN[1] > 1) {
                otherIdx += i3 * this->otherStrides[1];
            }
            if (this->otherN[2] > 1) {
                otherIdx += i2 * this->otherStrides[2];
            }
            if (this->otherN[3] > 1) {
                otherIdx += i1 * this->otherStrides[3];
            }

            const uint32_t sv = HalfToAbsInt(this->selfGm.GetValue(selfIdx));
            const uint32_t ov = HalfToAbsInt(this->otherGm.GetValue(otherIdx));
            const uint32_t g = EuclidGcd(sv, ov);
            this->outGm.SetValue(lin, static_cast<half>(g));
        }
    }
};

} // namespace

extern "C" __global__ __aicore__ void
gcd123(GM_ADDR self, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGcd123 op;
    op.Init(self, other, out, tiling_data);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// CPU-side launcher used by the host runtime to dispatch the kernel.
void gcd123_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *self,
               uint8_t *other, uint8_t *out, uint8_t *workspace, uint8_t *tiling)
{
    gcd123<<<blockDim, l2ctrl, stream>>>(self, other, out, workspace, tiling);
}
#endif
