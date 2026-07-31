#include "kernel_operator.h"
#include "softshrink_grad_tiling.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t FP32_VECTOR_ALIGN_ELEMENTS = 64;
}  // namespace

template <typename T>
class KernelSoftshrinkGrad {
public:
    __aicore__ inline KernelSoftshrinkGrad() {}

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                const SoftshrinkGradTilingData &tiling)
    {
        totalLength_ = tiling.totalLength;
        blockLength_ = tiling.blockLength;
        tileLength_ = tiling.tileLength;
        lambd_ = tiling.lambd;

        const uint64_t blockStart =
            static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockLength_;
        if (blockStart >= totalLength_ || tileLength_ == 0) {
            valid_ = false;
            return;
        }

        blockStart_ = blockStart;
        const uint64_t remaining = totalLength_ - blockStart_;
        actualBlockLength_ = static_cast<uint32_t>(
            remaining < blockLength_ ? remaining : blockLength_);
        gradOutputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(gradOutput) + blockStart_, actualBlockLength_);
        selfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(self) + blockStart_, actualBlockLength_);
        gradInputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(gradInput) + blockStart_, actualBlockLength_);

        pipe_.InitBuffer(gradQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(selfQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(gradFp32Buffer_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(selfFp32Buffer_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outputFp32Buffer_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(absBuffer_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(maskBuffer_, tileLength_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        const uint32_t loopCount =
            (actualBlockLength_ + tileLength_ - 1) / tileLength_;
        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            const uint32_t offset = loop * tileLength_;
            const uint32_t remaining = actualBlockLength_ - offset;
            const uint32_t validLength =
                remaining < tileLength_ ? remaining : tileLength_;
            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t validLength)
    {
        AscendC::LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        AscendC::LocalTensor<T> selfLocal = selfQueue_.AllocTensor<T>();
        const AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(gradLocal, gradOutputGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(selfLocal, selfGm_[offset], copyParams, padParams);
        gradQueue_.EnQue(gradLocal);
        selfQueue_.EnQue(selfLocal);
    }

    __aicore__ inline void Compute(uint32_t validLength)
    {
        AscendC::LocalTensor<T> gradLocal = gradQueue_.DeQue<T>();
        AscendC::LocalTensor<T> selfLocal = selfQueue_.DeQue<T>();
        AscendC::LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        AscendC::LocalTensor<uint8_t> maskLocal = maskBuffer_.Get<uint8_t>();
        const uint32_t computeLength =
            (validLength + FP32_VECTOR_ALIGN_ELEMENTS - 1) /
            FP32_VECTOR_ALIGN_ELEMENTS * FP32_VECTOR_ALIGN_ELEMENTS;

        if constexpr (AscendC::IsSameType<T, float>::value) {
            AscendC::LocalTensor<float> absLocal = absBuffer_.Get<float>();
            AscendC::Abs(absLocal, selfLocal, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(maskLocal, absLocal, lambd_,
                                   AscendC::CMPMODE::GT, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(outputLocal, maskLocal, gradLocal, 0.0F,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            computeLength);
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradFp32Buffer_.Get<float>();
            AscendC::LocalTensor<float> selfFp32 = selfFp32Buffer_.Get<float>();
            AscendC::LocalTensor<float> outputFp32 = outputFp32Buffer_.Get<float>();
            AscendC::LocalTensor<float> absLocal = absBuffer_.Get<float>();
            AscendC::Cast(gradFp32, gradLocal, AscendC::RoundMode::CAST_NONE,
                          computeLength);
            AscendC::Cast(selfFp32, selfLocal, AscendC::RoundMode::CAST_NONE,
                          computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(absLocal, selfFp32, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(maskLocal, absLocal, lambd_,
                                   AscendC::CMPMODE::GT, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(outputFp32, maskLocal, gradFp32, 0.0F,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                AscendC::Cast(outputLocal, outputFp32,
                              AscendC::RoundMode::CAST_RINT, computeLength);
            } else {
                AscendC::Cast(outputLocal, outputFp32,
                              AscendC::RoundMode::CAST_NONE, computeLength);
            }
        }

        outputQueue_.EnQue(outputLocal);
        gradQueue_.FreeTensor(gradLocal);
        selfQueue_.FreeTensor(selfLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t validLength)
    {
        AscendC::LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gradInputGm_[offset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gradQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> selfQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradFp32Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfFp32Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputFp32Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> absBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuffer_;
    AscendC::GlobalTensor<T> gradOutputGm_;
    AscendC::GlobalTensor<T> selfGm_;
    AscendC::GlobalTensor<T> gradInputGm_;
    uint64_t totalLength_ = 0;
    uint64_t blockStart_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t actualBlockLength_ = 0;
    uint32_t tileLength_ = 0;
    float lambd_ = 0.5F;
    bool valid_ = true;
};

extern "C" __global__ __aicore__ void softshrink_grad(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftshrinkGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftshrinkGradTilingData, tilingData, tiling);
    KernelSoftshrinkGrad<DTYPE_GRADOUTPUT> op;
    op.Init(gradOutput, self, gradInput, tilingData);
    op.Process();
}
