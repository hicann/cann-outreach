#include "kernel_operator.h"

#include <type_traits>

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr float TAIL_LIMIT = 16.0f;
constexpr uint32_t FMA_MIN_CORE_ELEMENTS = 1024;
constexpr uint32_t COEFF_BLOCK_ELEMENTS = DATA_BLOCK_BYTES / sizeof(float);
// Ascending coefficients for 0.5*erfcx(a/sqrt(2)) ~= P4(a)/Q5(a).
constexpr float P0 = 0.5f;
constexpr float P1 = 0.4437195360660553f;
constexpr float P2 = 0.18732434511184692f;
constexpr float P3 = 0.04203106835484505f;
constexpr float P4 = 0.004328861832618713f;
constexpr float Q0 = 1.0f;
constexpr float Q1 = 1.6853243112564087f;
constexpr float Q2 = 1.2193360328674316f;
constexpr float Q3 = 0.480280339717865f;
constexpr float Q4 = 0.10536123067140579f;
constexpr float Q5 = 0.010850762017071247f;
}  // namespace

template <typename DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR output, uint64_t totalLength,
                                uint64_t blockLength, uint32_t tileLength)
    {
        this->tileLength = tileLength;

        const uint64_t blockOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockLength;
        if (blockOffset >= totalLength) {
            coreLength = 0;
        } else {
            const uint64_t remaining = totalLength - blockOffset;
            coreLength = remaining < blockLength ? remaining : blockLength;
        }

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX) + blockOffset, coreLength);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output) + blockOffset, coreLength);

        // A one-tile core cannot overlap different tiles. Allocate a second
        // queue slot only when the core actually has more work to prefetch.
        const uint32_t bufferCount = coreLength > tileLength ? BUFFER_NUM : 1;
        pipe.InitBuffer(inputQueue, bufferCount, tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueue, bufferCount, tileLength * sizeof(DT_INPUT_X));

        // Four fp32 vectors: |x|, denominator, Gaussian, numerator.
        // The half path needs one more fp32 vector for the input/result.
        pipe.InitBuffer(absBuffer, tileLength * sizeof(float));
        pipe.InitBuffer(transformBuffer, tileLength * sizeof(float));
        pipe.InitBuffer(gaussianBuffer, tileLength * sizeof(float));
        pipe.InitBuffer(polynomialBuffer, tileLength * sizeof(float));
        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            pipe.InitBuffer(floatBuffer, tileLength * sizeof(float));
        }
        useFma = coreLength >= FMA_MIN_CORE_ELEMENTS;
        if (useFma) {
            // Each constant occupies one 32-byte block, broadcast by stride 0.
            // Initialize once per core, not once per tile. The host's 8 KiB
            // reserve covers these additional 192 bytes.
            pipe.InitBuffer(coefficientBuffer, 6 * DATA_BLOCK_BYTES);
            auto c = coefficientBuffer.Get<float>();
            AscendC::Duplicate(c[0 * COEFF_BLOCK_ELEMENTS], P2, COEFF_BLOCK_ELEMENTS);
            AscendC::Duplicate(c[1 * COEFF_BLOCK_ELEMENTS], Q2, COEFF_BLOCK_ELEMENTS);
            AscendC::Duplicate(c[2 * COEFF_BLOCK_ELEMENTS], P1, COEFF_BLOCK_ELEMENTS);
            AscendC::Duplicate(c[3 * COEFF_BLOCK_ELEMENTS], Q1, COEFF_BLOCK_ELEMENTS);
            AscendC::Duplicate(c[4 * COEFF_BLOCK_ELEMENTS], P0, COEFF_BLOCK_ELEMENTS);
            AscendC::Duplicate(c[5 * COEFF_BLOCK_ELEMENTS], Q0, COEFF_BLOCK_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0) {
            return;
        }

        // Short inputs bypass the 64-bit divide/modulo and prefetch loop.
        if (coreLength <= tileLength) {
            const uint32_t count = static_cast<uint32_t>(coreLength);
            CopyIn<false>(0, count);
            Compute(count);
            CopyOut<false>(0, count);
            return;
        }
        const uint64_t fullTiles = coreLength / tileLength;
        const uint32_t tailCount = static_cast<uint32_t>(coreLength % tileLength);

        CopyIn<true>(0, tileLength);
        uint64_t offset = 0;
        for (uint64_t i = 0; i < fullTiles; ++i, offset += tileLength) {
            // Prefetch exactly one tile. Full tiles have no per-copy alignment
            // branch; only the final partial tile needs the general copy path.
            if (i + 1 < fullTiles) {
                CopyIn<true>(offset + tileLength, tileLength);
            } else if (tailCount != 0) {
                CopyIn<false>(offset + tileLength, tailCount);
            }
            Compute(tileLength);
            CopyOut<true>(offset, tileLength);
        }
        if (tailCount != 0) {
            Compute(tailCount);
            CopyOut<false>(offset, tailCount);
        }
    }

private:
    template <bool FULL_TILE>
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.AllocTensor<DT_INPUT_X>();
        if constexpr (FULL_TILE) {
            AscendC::DataCopy(inputLocal, inputGm[offset], count);
        } else if ((count * sizeof(DT_INPUT_X)) % DATA_BLOCK_BYTES == 0) {
            AscendC::DataCopy(inputLocal, inputGm[offset], count);
        } else {
            // DataCopyPad is required for the final tile when its byte length is
            // not a multiple of 32.  Padding remains in UB and is never written.
            AscendC::DataCopyParams copyParams{
                1, static_cast<uint16_t>(count * sizeof(DT_INPUT_X)), 0, 0};
            AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
        }
        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.AllocTensor<DT_INPUT_X>();
        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            AscendC::LocalTensor<float> floatLocal = floatBuffer.Get<float>();
            AscendC::Cast(floatLocal, inputLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeGelu(floatLocal, floatLocal, count);
            AscendC::Cast(outputLocal, floatLocal, AscendC::RoundMode::CAST_RINT, count);
        } else {
            ComputeGelu(outputLocal, inputLocal, count);
        }

        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    template <bool USE_FMA, uint32_t COEFF_INDEX>
    __aicore__ inline void HornerPair(const AscendC::LocalTensor<float> &p,
                                      const AscendC::LocalTensor<float> &q,
                                      const AscendC::LocalTensor<float> &a,
                                      float pc, float qc)
    {
        constexpr uint64_t mask = AscendC::MASK_PLACEHOLDER;
        if constexpr (USE_FMA) {
            auto c = coefficientBuffer.Get<float>();
            // FusedMulAdd implements dst = src0 * dst + src1.
            const AscendC::BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
            AscendC::FusedMulAdd<float, false>(p, a,
                c[COEFF_INDEX * COEFF_BLOCK_ELEMENTS], mask, 1, broadcast);
            AscendC::FusedMulAdd<float, false>(q, a,
                c[(COEFF_INDEX + 1) * COEFF_BLOCK_ELEMENTS], mask, 1, broadcast);
        } else {
            const AscendC::BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
            const AscendC::UnaryRepeatParams unary{1, 1, 8, 8};
            AscendC::Mul<float, false>(p, p, a, mask, 1, binary);
            AscendC::Mul<float, false>(q, q, a, mask, 1, binary);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float, false>(p, p, pc, mask, 1, unary);
            AscendC::Adds<float, false>(q, q, qc, mask, 1, unary);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeGelu(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src, uint32_t count)
    {
        if (useFma) {
            ComputeGeluImpl<true>(dst, src, count);
        } else {
            ComputeGeluImpl<false>(dst, src, count);
        }
    }

    template <bool USE_FMA>
    __aicore__ inline void ComputeGeluImpl(const AscendC::LocalTensor<float> &dst,
                                          const AscendC::LocalTensor<float> &src, uint32_t count)
    {
        AscendC::LocalTensor<float> a = absBuffer.Get<float>();
        AscendC::LocalTensor<float> q = transformBuffer.Get<float>();
        AscendC::LocalTensor<float> e = gaussianBuffer.Get<float>();
        AscendC::LocalTensor<float> p = polynomialBuffer.Get<float>();
        const AscendC::UnaryRepeatParams unary{1, 1, 8, 8};
        const AscendC::BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        constexpr uint64_t mask = AscendC::MASK_PLACEHOLDER;

        // GELU(x) = max(x,0) - |x|*Phi(-|x|).
        // Phi(-a) ~= exp(-a*a/2) * P4(a)/Q5(a), a in [0,16].
        // All coefficients are positive, avoiding polynomial cancellation.
        // Independent P/Q chains share barriers and overlap the Gaussian work.
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(0, count);
        AscendC::Abs<float, false>(a, src, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();
        // src may alias dst (fp16 path), hence the barrier after Abs.
        AscendC::Maxs<float, false>(dst, src, 0.0f, mask, 1, unary);
        AscendC::Mins<float, false>(a, a, TAIL_LIMIT, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();

        // Clamp before squaring so the full finite fp32 input range is safe.
        // At a=16 the Gaussian is already below fp32's subnormal range.
        AscendC::Mul<float, false>(e, a, a, mask, 1, binary);
        AscendC::Muls<float, false>(q, a, Q5, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float, false>(e, e, -0.5f, mask, 1, unary);
        AscendC::Adds<float, false>(q, q, Q4, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(e, e, mask, 1, unary);
        AscendC::Mul<float, false>(q, q, a, mask, 1, binary);
        AscendC::Muls<float, false>(p, a, P4, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<float, false>(e, e, a, mask, 1, binary);
        AscendC::Adds<float, false>(q, q, Q3, mask, 1, unary);
        AscendC::Adds<float, false>(p, p, P3, mask, 1, unary);
        AscendC::PipeBarrier<PIPE_V>();

        HornerPair<USE_FMA, 0>(p, q, a, P2, Q2);
        HornerPair<USE_FMA, 2>(p, q, a, P1, Q1);
        HornerPair<USE_FMA, 4>(p, q, a, P0, Q0);
        // Keep Div for precision; Q5(a) >= 1 throughout the clamped domain.
        AscendC::Div<float, false>(p, p, q, mask, 1, binary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<float, false>(e, e, p, mask, 1, binary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(dst, dst, e, mask, 1, binary);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    template <bool FULL_TILE>
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.DeQue<DT_INPUT_X>();
        if constexpr (FULL_TILE) {
            AscendC::DataCopy(outputGm[offset], outputLocal, count);
        } else if ((count * sizeof(DT_INPUT_X)) % DATA_BLOCK_BYTES == 0) {
            AscendC::DataCopy(outputGm[offset], outputLocal, count);
        } else {
            AscendC::DataCopyParams copyParams{
                1, static_cast<uint16_t>(count * sizeof(DT_INPUT_X)), 0, 0};
            AscendC::DataCopyPad(outputGm[offset], outputLocal, copyParams);
        }
        outputQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> absBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transformBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gaussianBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> polynomialBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> coefficientBuffer;
    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;
    uint64_t coreLength = 0;
    uint32_t tileLength = 0;
    bool useFma = false;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
