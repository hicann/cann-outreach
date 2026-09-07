// GELU kernel: y = 0.5 * x * (1 + erf(x / sqrt(2))).
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr float INV_SQRT2 = 0.70710678118654752440f;
constexpr float HALF_SCALAR = 0.5f;
constexpr float ONE_SCALAR = 1.0f;
}  // namespace

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                uint64_t totalLength, uint64_t blockLength,
                                uint32_t tileLength)
    {
        this->tileLength = tileLength;
        const uint64_t blockOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockLength;

        if (blockOffset >= totalLength) {
            this->coreLength = 0;
            return;
        }

        const uint64_t remaining = totalLength - blockOffset;
        this->coreLength = remaining < blockLength ? remaining : blockLength;

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + blockOffset, this->coreLength);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output) + blockOffset, this->coreLength);

        pipe.InitBuffer(inputQueue, 1, this->tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueue, 1, this->tileLength * sizeof(DT_INPUT_X));

        // float32 intermediates are intentionally used for fp16 input.  This
        // avoids accumulating half-precision rounding error and comfortably
        // meets the required 1e-3 tolerance after the final fp16 cast.
        pipe.InitBuffer(workBuf0, this->tileLength * sizeof(float));
        if constexpr (AscendC::IsSameType<DT_INPUT_X, half>::value) {
            pipe.InitBuffer(workBuf1, this->tileLength * sizeof(float));
            pipe.InitBuffer(workBuf2, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0) {
            return;
        }

        uint64_t offset = 0;
        while (offset < this->coreLength) {
            const uint64_t remain = this->coreLength - offset;
            const uint32_t count = static_cast<uint32_t>(remain < this->tileLength ? remain : this->tileLength);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline uint32_t AlignCount(uint32_t count) const
    {
        constexpr uint32_t elemPerBlock = BYTE_BLOCK / sizeof(DT_INPUT_X);
        return ((count + elemPerBlock - 1) / elemPerBlock) * elemPerBlock;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.template AllocTensor<DT_INPUT_X>();

        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignCount(count) - count;
        padParams.paddingValue = static_cast<DT_INPUT_X>(0);
        AscendC::DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);

        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.template DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.template AllocTensor<DT_INPUT_X>();

        if constexpr (AscendC::IsSameType<DT_INPUT_X, float>::value) {
            AscendC::LocalTensor<float> scaled = workBuf0.template Get<float>();

            AscendC::Muls(scaled, inputLocal, INV_SQRT2, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Erf<float, false>(outputLocal, scaled, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(outputLocal, outputLocal, ONE_SCALAR, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(outputLocal, outputLocal, HALF_SCALAR, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outputLocal, outputLocal, inputLocal, count);
        } else {
            AscendC::LocalTensor<float> inputFloat = workBuf0.template Get<float>();
            AscendC::LocalTensor<float> scaled = workBuf1.template Get<float>();
            AscendC::LocalTensor<float> resultFloat = workBuf2.template Get<float>();

            AscendC::Cast(inputFloat, inputLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scaled, inputFloat, INV_SQRT2, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Erf<float, false>(resultFloat, scaled, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(resultFloat, resultFloat, ONE_SCALAR, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(resultFloat, resultFloat, HALF_SCALAR, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(resultFloat, resultFloat, inputFloat, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, resultFloat, AscendC::RoundMode::CAST_NONE, count);
        }

        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.template DeQue<DT_INPUT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[offset], outputLocal, copyParams);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf2;
    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;
    uint64_t coreLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);
    op.Process();
}
