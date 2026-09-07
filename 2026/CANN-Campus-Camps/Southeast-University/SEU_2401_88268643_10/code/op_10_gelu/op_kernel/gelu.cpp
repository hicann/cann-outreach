#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_BLOCK_BYTES = 32;

constexpr float GELU_APPROX_A = 1.59475758f;
constexpr float GELU_APPROX_B = 0.0916309959f;
constexpr float GELU_APPROX_C = 0.0108398377f;
constexpr float GELU_APPROX_CLAMP = 16.0f;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(
        GM_ADDR inputX, GM_ADDR output, const GeluTilingData* tilingData)
    {
        totalLength_ = tilingData->totalLength;
        tileLength_ = tilingData->tileLength;

        const uint64_t blockIndex = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t assignedLength;
        uint64_t blockOffset;
        if (blockIndex < tilingData->largeCoreCount) {
            assignedLength = tilingData->largeBlockLength;
            blockOffset = blockIndex * tilingData->largeBlockLength;
        } else {
            assignedLength = tilingData->smallBlockLength;
            blockOffset =
                static_cast<uint64_t>(tilingData->largeCoreCount) *
                    tilingData->largeBlockLength +
                (blockIndex - tilingData->largeCoreCount) *
                    tilingData->smallBlockLength;
        }
        if (blockOffset >= totalLength_) {
            blockLength_ = 0;
            return;
        }

        const uint64_t remaining = totalLength_ - blockOffset;
        blockLength_ = remaining < assignedLength
            ? remaining
            : assignedLength;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(inputX) + blockOffset, blockLength_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output) + blockOffset, blockLength_);

        const uint8_t bufferNum = blockLength_ <= tileLength_ ? 1 : BUFFER_NUM;
        const uint32_t tileBytes = tileLength_ * sizeof(T);
        pipe_.InitBuffer(inputQueue_, bufferNum, tileBytes);
        pipe_.InitBuffer(outputQueue_, bufferNum, tileBytes);
        if constexpr (AscendC::IsSameType<T, float>::value) {
            pipe_.InitBuffer(workBuffer_, tileBytes);
            pipe_.InitBuffer(workBuffer2_, tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }

        uint64_t offset = 0;
        while (offset < blockLength_) {
            const uint64_t remaining = blockLength_ - offset;
            const uint32_t count = static_cast<uint32_t>(
                remaining < tileLength_ ? remaining : tileLength_);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline bool IsAligned(uint32_t count) const
    {
        return (count * sizeof(T)) % DATA_BLOCK_BYTES == 0;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        if (IsAligned(count)) {
            AscendC::DataCopy(inputLocal, inputGm_[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(inputLocal, inputGm_[offset], copyParams, padParams);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
        AscendC::LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();

        if constexpr (!AscendC::IsSameType<T, float>::value) {
            // FP16's tolerance permits the substantially faster native GELU
            // path and avoids two casts plus the FP32 temporary tensors.
            AscendC::Gelu<T, false>(outputLocal, inputLocal, count);
        } else {
            AscendC::LocalTensor<float> clamped = workBuffer_.Get<float>();
            AscendC::LocalTensor<float> square = workBuffer2_.Get<float>();

            AscendC::Maxs(clamped, inputLocal, -GELU_APPROX_CLAMP, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(clamped, clamped, GELU_APPROX_CLAMP, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(square, clamped, clamped, count);
            AscendC::PipeBarrier<PIPE_V>();

            // numerator = A + B*x^2
            AscendC::Muls(outputLocal, square, GELU_APPROX_B, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(outputLocal, outputLocal, GELU_APPROX_A, count);

            // denominator = 1 + C*x^2
            AscendC::Muls(square, square, GELU_APPROX_C, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(square, square, 1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();

            // output = x / (1 + exp(-x * numerator / denominator))
            AscendC::Div(outputLocal, outputLocal, square, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outputLocal, outputLocal, clamped, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(outputLocal, outputLocal, -1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(outputLocal, outputLocal, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(outputLocal, outputLocal, 1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(outputLocal, inputLocal, outputLocal, count);
        }

        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
        if (IsAligned(count)) {
            AscendC::DataCopy(outputGm_[offset], outputLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuffer2_;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    uint64_t totalLength_ = 0;
    uint64_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

} // namespace

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, &tilingData);
    op.Process();
}
