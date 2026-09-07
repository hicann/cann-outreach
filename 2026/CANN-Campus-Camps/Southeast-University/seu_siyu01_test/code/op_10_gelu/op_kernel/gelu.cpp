#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <typename T, uint32_t BufferCount>
class KernelGelu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const GeluTilingData& data) {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), data.length);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), data.length);
        tileElements_ = data.tile_elements;
        const uint32_t block = GetBlockIdx();
        if (block >= data.block_count || data.length == 0) {
            return;
        }
        // Host precomputes the quotient and remainder of aligned core partitioning.
        // The device only performs multiplication, addition, and comparisons.
        constexpr uint32_t kAlign = 32 / sizeof(T);
        const uint32_t precedingExtra = block < data.extra_core_count ? block : data.extra_core_count;
        begin_ = data.core_elements * block + static_cast<uint64_t>(precedingExtra) * kAlign;
        if (begin_ >= data.length) {
            return;
        }
        const uint64_t assigned = data.core_elements + (block < data.extra_core_count ? kAlign : 0);
        count_ = assigned < data.length - begin_ ? assigned : data.length - begin_;
        pipe_.InitBuffer(inputQueue_, BufferCount, tileElements_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BufferCount, tileElements_ * sizeof(T));
        pipe_.InitBuffer(argBuffer_, tileElements_ * sizeof(float));
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe_.InitBuffer(xBuffer_, tileElements_ * sizeof(float));
            pipe_.InitBuffer(valueBuffer_, tileElements_ * sizeof(float));
        }
        pipe_.InitBuffer(tBuffer_, tileElements_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (count_ == 0 || tileElements_ == 0) {
            return;
        }
        if constexpr (BufferCount == 1) {
            // Host selects this specialization only when the entire core fits in one tile.
            const uint32_t valid = static_cast<uint32_t>(count_);
            CopyIn(begin_, valid);
            Compute(valid);
            CopyOut(begin_, valid);
        } else {
            uint64_t offset = begin_;
            uint64_t remaining = count_;
            CopyIn(offset, TileCount(remaining));
            while (remaining != 0) {
                const uint32_t valid = TileCount(remaining);
                if (remaining > valid) {
                    CopyIn(offset + valid, TileCount(remaining - valid));
                }
                Compute(valid);
                CopyOut(offset, valid);
                offset += valid;
                remaining -= valid;
            }
        }
    }

private:
    __aicore__ inline uint32_t TileCount(uint64_t remaining) const {
        return static_cast<uint32_t>(remaining < tileElements_ ? remaining : tileElements_);
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t valid) {
        LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        const uint32_t bytes = valid * sizeof(T);
        if (bytes % 32 == 0) {
            DataCopy(input, inputGm_[offset], valid);
        } else {
            const DataCopyExtParams params{1, bytes, 0, 0, 0};
            const DataCopyPadExtParams<T> padding{true, 0, 0, T(0)};
            DataCopyPad(input, inputGm_[offset], params, padding);
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Evaluate(const LocalTensor<float>& output,
                                    const LocalTensor<float>& input,
                                    const LocalTensor<float>& argument,
                                    uint32_t valid) {
        // Fused GELU = max(x,0) - a*phi(a)*R(a), a=|x|.
        // The direct-t polynomial includes the normal density constant.
        // t=1/(1+0.2*a). This is the degree-6 V5 fit after substituting u=2*t-1.
        // Clipping the correction argument prevents overflow for finite FLT_MAX;
        // exp(-128) rounds to zero in float32, yielding the correct saturated tail.
        // Standard custom-op builds enable --cce-auto-sync. All vector dependencies
        // below use direct LocalTensor references, with no address reinterpretation.
        LocalTensor<float> t = tBuffer_.Get<float>();
        Abs(argument, input, valid);
        Mins(argument, argument, 16.0f, valid);
        Muls(t, argument, 0.2f, valid);
        Adds(t, t, 1.0f, valid);
        Duplicate(output, 1.0f, valid);
        Div(t, output, t, valid);
        Muls(output, t, 0.2771728635f, valid);
        Adds(output, output, -0.3798909783f, valid);
        Mul(output, output, t, valid);
        Adds(output, output, 0.4368641376f, valid);
        Mul(output, output, t, valid);
        Adds(output, output, -0.1156639159f, valid);
        Mul(output, output, t, valid);
        Adds(output, output, 0.1294990927f, valid);
        Mul(output, output, t, valid);
        Adds(output, output, 0.07172055542f, valid);
        Mul(output, output, t, valid);
        Adds(output, output, 0.08029820025f, valid);
        Mul(output, output, t, valid);
        // Reciprocal t is now dead; reuse the same buffer for a*exp(-a*a/2).
        Mul(t, argument, argument, valid);
        Muls(t, t, -0.5f, valid);
        Exp(t, t, valid);
        Mul(t, t, argument, valid);
        Mul(output, output, t, valid);
        Maxs(t, input, 0.0f, valid);
        Sub(output, t, output, valid);
    }

    __aicore__ inline void Compute(uint32_t valid) {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        LocalTensor<float> argument = argBuffer_.Get<float>();
        if constexpr (sizeof(T) == sizeof(float)) {
            // No UB-to-UB copies: read VECIN directly and write VECOUT directly.
            Evaluate(output, input, argument, valid);
        } else {
            LocalTensor<float> x = xBuffer_.Get<float>();
            LocalTensor<float> value = valueBuffer_.Get<float>();
            Cast(x, input, RoundMode::CAST_NONE, valid);
            PipeBarrier<PIPE_V>();
            Evaluate(value, x, argument, valid);
            PipeBarrier<PIPE_V>();
            Cast(output, value, RoundMode::CAST_RINT, valid);
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t valid) {
        LocalTensor<T> output = outputQueue_.DeQue<T>();
        const uint32_t bytes = valid * sizeof(T);
        if (bytes % 32 == 0) {
            DataCopy(outputGm_[offset], output, valid);
        } else {
            DataCopyPad(outputGm_[offset], output, DataCopyExtParams{1, bytes, 0, 0, 0});
        }
        outputQueue_.FreeTensor(output);
    }

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, BufferCount> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> argBuffer_;
    TBuf<TPosition::VECCALC> xBuffer_;
    TBuf<TPosition::VECCALC> valueBuffer_;
    TBuf<TPosition::VECCALC> tBuffer_;
    uint64_t begin_ = 0;
    uint64_t count_ = 0;
    uint32_t tileElements_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, data, tiling);
    if (data.buffer_count == 1) {
        KernelGelu<DT_INPUT_X, 1> op;
        op.Init(input_x, output, data);
        op.Process();
    } else {
        KernelGelu<DT_INPUT_X, 2> op;
        op.Init(input_x, output, data);
        op.Process();
    }
}

