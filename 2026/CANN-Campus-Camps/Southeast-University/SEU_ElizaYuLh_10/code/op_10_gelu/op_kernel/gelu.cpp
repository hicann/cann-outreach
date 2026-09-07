// Kernel-side GELU implementation for Ascend 910B.
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() = default;

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        constexpr uint32_t kBlockElements = 64;  // 32-byte DMA block for FP16/FP32.

        const uint32_t block_count =
            (length + kBlockElements - 1) / kBlockElements;
        const uint32_t core_count = GetBlockNum();
        const uint32_t core_index = GetBlockIdx();
        if (core_count == 0 || core_index >= core_count || block_count == 0) {
            count_ = 0;
            return;
        }

        // Balanced partition in 32-byte DMA blocks. At most one core owns a
        // partial final block, so every core starts at a DMA-aligned address.
        const uint32_t base_blocks = block_count / core_count;
        const uint32_t extra_blocks = block_count % core_count;
        const uint32_t start_blocks =
            core_index * base_blocks + (core_index < extra_blocks ? core_index : extra_blocks);
        const uint32_t assigned_blocks =
            base_blocks + (core_index < extra_blocks ? 1 : 0);
        const uint32_t start = start_blocks * kBlockElements;
        const uint32_t assigned = assigned_blocks * kBlockElements;

        if (start >= length) {
            count_ = 0;
            return;
        }

        count_ = length - start;
        if (count_ > assigned) {
            count_ = assigned;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + start, count_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output) + start, count_);

        // FP32 uses three FP32 calculation buffers, while FP16 needs an extra
        // FP32 buffer for conversion. A single output buffer is sufficient
        // because output is consumed immediately; this frees UB for a larger
        // FP16 tile without changing numerical behavior.
        constexpr uint32_t kTile = TileSize();
        const uint32_t capacity =
            count_ < kTile ? ((count_ + kBlockElements - 1) / kBlockElements) * kBlockElements : kTile;
        const uint32_t input_buffers = count_ > kTile ? 2 : 1;

        pipe_.InitBuffer(inQueue_, input_buffers, capacity * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueue_, 1, capacity * sizeof(DT_INPUT_X));
        if constexpr (sizeof(DT_INPUT_X) != sizeof(float)) {
            pipe_.InitBuffer(xBuf_, capacity * sizeof(float));
        }
        pipe_.InitBuffer(aBuf_, capacity * sizeof(float));
        pipe_.InitBuffer(zBuf_, capacity * sizeof(float));
        pipe_.InitBuffer(pBuf_, capacity * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (count_ == 0) {
            return;
        }

        constexpr uint32_t kTile = TileSize();
        uint32_t current_offset = 0;
        uint32_t current_count = count_ < kTile ? count_ : kTile;

        CopyIn(current_offset, current_count);

        while (current_offset < count_) {
            const uint32_t next_offset = current_offset + current_count;
            if (next_offset < count_) {
                const uint32_t next_count =
                    (count_ - next_offset) < kTile ? (count_ - next_offset) : kTile;
                CopyIn(next_offset, next_count);
            }

            Compute(current_count);
            CopyOut(current_offset, current_count);

            current_offset = next_offset;
            if (current_offset < count_) {
                current_count = (count_ - current_offset) < kTile
                    ? (count_ - current_offset)
                    : kTile;
            }
        }
    }

private:
    static constexpr uint32_t TileSize() {
        // FP32 already reaches the 112 KiB UB budget with a 4096-element
        // double-buffered input path. FP16 can use 5120 elements with a
        // single output buffer, reducing DMA/queue overhead for long tensors.
        return sizeof(DT_INPUT_X) == sizeof(float) ? 4096u : 5120u;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t n) {
        LocalTensor<DT_INPUT_X> input = inQueue_.AllocTensor<DT_INPUT_X>();
        constexpr uint32_t kDmaBlockElements = 32 / sizeof(DT_INPUT_X);

        if (n % kDmaBlockElements == 0) {
            DataCopy(input, xGm_[offset], n);
        } else {
            const DataCopyExtParams copy = {
                1, n * static_cast<uint32_t>(sizeof(DT_INPUT_X)), 0, 0, 0};
            const DataCopyPadExtParams<DT_INPUT_X> pad = {false, 0, 0, 0};
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
            // Reuse queue storage directly for FP32: no conversion/copy buffer.
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

        // GELU(x) = max(x, 0) - |x| * sigmoid(q(|x|)), where q is an odd
        // polynomial approximation of log(Phi(-a) / Phi(a)) / a.
        // Clamping a to 6 keeps the polynomial in its fitted range; for
        // |x| > 6 the correction is already negligible in FP32/FP16 output.
        Abs(a, x, n);
        Mins(a, a, 6.0f, n);
        Mul(z, a, a, n);

        if constexpr (sizeof(DT_INPUT_X) == sizeof(float)) {
            // Degree-4 polynomial: max error remains in the low-e-6 range.
            Muls(p, z, -2.225263187198141e-6f, n);
            Adds(p, p, 6.093439338766565e-5f, n);
            Mul(p, p, z, n);
            Adds(p, p, 2.5095241553991783e-4f, n);
            Mul(p, p, z, n);
            Adds(p, p, -0.07293986152942938f, n);
            Mul(p, p, z, n);
            Adds(p, p, -1.5956551532991676f, n);
        } else {
            // Lower-order approximation is sufficient after FP16 output
            // rounding and saves vector instructions.
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
        constexpr uint32_t kDmaBlockElements = 32 / sizeof(DT_INPUT_X);

        if (n % kDmaBlockElements == 0) {
            DataCopy(yGm_[offset], output, n);
        } else {
            const DataCopyExtParams copy = {
                1, n * static_cast<uint32_t>(sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], output, copy);
        }
        outQueue_.FreeTensor(output);
    }

    uint32_t count_ = 0;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<QuePosition::VECCALC> xBuf_, aBuf_, zBuf_, pBuf_;
    GlobalTensor<DT_INPUT_X> xGm_, yGm_;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}
