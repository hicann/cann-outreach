// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length,
        uint32_t blockLength, uint32_t tileLength) {
        length_ = length;
        blockLength_ = blockLength;
        tileLength_ = tileLength;
        blockOffset_ = static_cast<uint32_t>(AscendC::GetBlockIdx()) * blockLength_;
        if (blockOffset_ >= length_) {
            blockLength_ = 0;
            return;
        }
        uint32_t remaining = length_ - blockOffset_;
        if (blockLength_ > remaining) {
            blockLength_ = remaining;
        }
        inputGM_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + blockOffset_, blockLength_);
        outputGM_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + blockOffset_, blockLength_);
        pipe_.InitBuffer(inputQueue_, 2, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outputQueue_, 2, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(inputFloat_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(erfInput_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(resultFloat_, tileLength_ * sizeof(float));
    }
    __aicore__ inline void Process() {
        if (blockLength_ == 0) {
            return;
        }
        uint32_t offset = 0;
        while (offset < blockLength_) {
            uint32_t current = blockLength_ - offset;
            if (current > tileLength_) {
                current = tileLength_;
            }
            const uint32_t alignElements = 32 / sizeof(DT_INPUT_X);
            const uint32_t compute = (current + alignElements - 1) / alignElements * alignElements;
            AscendC::LocalTensor<DT_INPUT_X> x = inputQueue_.AllocTensor<DT_INPUT_X>();
            if (current % alignElements == 0) {
                AscendC::DataCopy(x, inputGM_[offset], current);
            } else {
                AscendC::DataCopyExtParams copyParams{
                    1, static_cast<uint32_t>(current * sizeof(DT_INPUT_X)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{};
                padParams.isPad = true;
                padParams.rightPadding = static_cast<uint8_t>(alignElements - current % alignElements);
                AscendC::DataCopyPad(x, inputGM_[offset], copyParams, padParams);
            }
            inputQueue_.EnQue(x);
            x = inputQueue_.DeQue<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> y = outputQueue_.AllocTensor<DT_INPUT_X>();
            ComputeGelu(y, x, compute);
            outputQueue_.EnQue(y);
            inputQueue_.FreeTensor(x);
            y = outputQueue_.DeQue<DT_INPUT_X>();
            if (current % alignElements == 0) {
                AscendC::DataCopy(outputGM_[offset], y, current);
            } else {
                AscendC::DataCopyExtParams copyParams{
                    1, static_cast<uint32_t>(current * sizeof(DT_INPUT_X)), 0, 0, 0};
                AscendC::DataCopyPad(outputGM_[offset], y, copyParams);
            }
            outputQueue_.FreeTensor(y);
            offset += current;
        }
    }
private:
    __aicore__ inline void ComputeGelu(
        AscendC::LocalTensor<DT_INPUT_X> y,
        AscendC::LocalTensor<DT_INPUT_X> x, uint32_t count) {
        auto xf = inputFloat_.Get<float>();
        auto z = erfInput_.Get<float>();
        auto result = resultFloat_.Get<float>();
        if constexpr (sizeof(DT_INPUT_X) == sizeof(half)) {
            AscendC::Cast(xf, x, AscendC::RoundMode::CAST_NONE, count);
        } else {
            AscendC::Adds(xf, x, 0.0f, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Bound only the erf argument; preserve the original input for multiplication.
        AscendC::Mins(z, xf, 10.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(z, z, -10.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(z, z, 0.7071067811865475f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Erf(result, z, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(result, result, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(result, result, 0.5f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(result, xf, result, count);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (sizeof(DT_INPUT_X) == sizeof(half)) {
            AscendC::Cast(y, result, AscendC::RoundMode::CAST_ROUND, count);
        } else {
            AscendC::Adds(y, result, 0.0f, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputFloat_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> erfInput_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultFloat_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputQueue_;
    AscendC::GlobalTensor<DT_INPUT_X> inputGM_;
    AscendC::GlobalTensor<DT_INPUT_X> outputGM_;
    uint32_t length_ = 0;
    uint32_t blockOffset_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 4096;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
