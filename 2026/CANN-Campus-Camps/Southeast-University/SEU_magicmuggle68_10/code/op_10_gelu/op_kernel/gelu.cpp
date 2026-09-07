// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {

constexpr uint32_t BUFFER_NUM = 2;
// 4096可进一步摊薄大张量的循环、队列和尾块开销；FP32双缓冲路径
// 连同两个计算缓冲区约占96KB UB，仍为910B保留充足余量。
constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DATA_BLOCK_BYTES = 32;

// Phi(x) = sigmoid(x * (A + B*x^2) / (1 + C*x^2)).
// 系数针对 x*Phi(x) 在float32下拟合，[-100, 100]密集抽样的最大绝对误差<3.6e-5。
constexpr float GELU_APPROX_A = 1.59475758f;
constexpr float GELU_APPROX_B = 0.0916309959f;
constexpr float GELU_APPROX_C = 0.0108398377f;
constexpr float GELU_APPROX_CLAMP = 16.0f;

}  // namespace

template <typename DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t blockOffset,
        uint32_t blockLength, AscendC::TPipe &pipe)
    {
        blockLength_ = blockLength;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + blockOffset, blockLength_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(output) + blockOffset, blockLength_);

        const uint8_t bufferNum = blockLength_ <= TILE_LENGTH ? 1 : BUFFER_NUM;
        const uint32_t tileBytes = TILE_LENGTH * sizeof(DT_INPUT_X);
        if constexpr (sizeof(DT_INPUT_X) == 2U) {
            // Gelu允许源、目的Tensor完全复用。绑定VECIN/VECOUT可省去一份FP16队列，
            // 同时把更多UB留给高阶API内部临时空间。
            pipe.InitBuffer(inOutQueue_, bufferNum, tileBytes);
        } else {
            pipe.InitBuffer(inputQueue_, bufferNum, tileBytes);
            pipe.InitBuffer(outputQueue_, bufferNum, tileBytes);
            pipe.InitBuffer(workBuffer_, tileBytes);
            pipe.InitBuffer(workBuffer2_, tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }

        const uint32_t loopCount = (blockLength_ + TILE_LENGTH - 1U) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            const uint32_t offset = i * TILE_LENGTH;
            uint32_t currentLength = blockLength_ - offset;
            if (currentLength > TILE_LENGTH) {
                currentLength = TILE_LENGTH;
            }
            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentLength)
    {
        if constexpr (sizeof(DT_INPUT_X) == 2U) {
            AscendC::LocalTensor<DT_INPUT_X> inputLocal =
                inOutQueue_.AllocTensor<DT_INPUT_X>();
            CopyInData(inputLocal, offset, currentLength);
            inOutQueue_.template EnQue<AscendC::TPosition::GM,
                AscendC::TPosition::VECIN, DT_INPUT_X>(inputLocal);
        } else {
            AscendC::LocalTensor<DT_INPUT_X> inputLocal =
                inputQueue_.AllocTensor<DT_INPUT_X>();
            CopyInData(inputLocal, offset, currentLength);
            inputQueue_.EnQue(inputLocal);
        }
    }

    __aicore__ inline void CopyInData(const AscendC::LocalTensor<DT_INPUT_X> &inputLocal,
        uint32_t offset, uint32_t currentLength)
    {
        const uint32_t copyBytes = currentLength * sizeof(DT_INPUT_X);
        if ((copyBytes & (DATA_BLOCK_BYTES - 1U)) == 0) {
            AscendC::DataCopy(inputLocal, inputGm_[offset], currentLength);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{
                false, 0, 0, static_cast<DT_INPUT_X>(0)};
            AscendC::DataCopyPad(inputLocal, inputGm_[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(uint32_t currentLength)
    {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
        if constexpr (sizeof(DT_INPUT_X) == 2U) {
            AscendC::LocalTensor<DT_INPUT_X> inOutLocal =
                inOutQueue_.template DeQue<AscendC::TPosition::GM,
                    AscendC::TPosition::VECIN, DT_INPUT_X>();
            // 默认模式已通过全部FP16精度测试；highPerformance模式会使测试点4
            // 出现1.18%的错误，因此这里保持精度可靠的默认实现。
            AscendC::Gelu<DT_INPUT_X, false>(inOutLocal, inOutLocal, currentLength);
            inOutQueue_.template EnQue<AscendC::TPosition::VECOUT,
                AscendC::TPosition::GM, DT_INPUT_X>(inOutLocal);
        } else {
            AscendC::LocalTensor<DT_INPUT_X> inputLocal =
                inputQueue_.DeQue<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> outputLocal =
                outputQueue_.AllocTensor<DT_INPUT_X>();
            AscendC::LocalTensor<float> clampedLocal = workBuffer_.Get<float>();
            AscendC::LocalTensor<float> squareLocal = workBuffer2_.Get<float>();

            // 全部Vector指令处理同一段连续数据，只设置一次Counter Mask，避免每条
            // tensor前N接口重复计算主块/尾块并切换Mask。
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLength);

            // 使用高精度有理逼辑函数近似Phi(x)，避免Erf高阶API的大量临时空间
            // 和内部分块开销。限制近似式输入可避免x^2溢出，且在区间外正确饱和到0/x。
            AscendC::Maxs<float, false>(clampedLocal, inputLocal, -GELU_APPROX_CLAMP,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins<float, false>(clampedLocal, clampedLocal, GELU_APPROX_CLAMP,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(squareLocal, clampedLocal, clampedLocal,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            // numerator = A + B*x^2
            AscendC::Muls<float, false>(outputLocal, squareLocal, GELU_APPROX_B,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float, false>(outputLocal, outputLocal, GELU_APPROX_A,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});

            // denominator = 1 + C*x^2
            AscendC::Muls<float, false>(squareLocal, squareLocal, GELU_APPROX_C,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float, false>(squareLocal, squareLocal, 1.0f,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            // output = x / (1 + exp(-x * numerator / denominator))
            AscendC::Div<float, false>(outputLocal, outputLocal, squareLocal,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(outputLocal, outputLocal, clampedLocal,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float, false>(outputLocal, outputLocal, -1.0f,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float, false>(outputLocal, outputLocal,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float, false>(outputLocal, outputLocal, 1.0f,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<float, false>(outputLocal, inputLocal, outputLocal,
                AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

            AscendC::SetMaskNorm();
            AscendC::ResetMask();

            outputQueue_.EnQue(outputLocal);
            inputQueue_.FreeTensor(inputLocal);
        }
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentLength)
    {
        if constexpr (sizeof(DT_INPUT_X) == 2U) {
            AscendC::LocalTensor<DT_INPUT_X> outputLocal =
                inOutQueue_.template DeQue<AscendC::TPosition::VECOUT,
                    AscendC::TPosition::GM, DT_INPUT_X>();
            CopyOutData(outputLocal, offset, currentLength);
            inOutQueue_.FreeTensor(outputLocal);
        } else {
            AscendC::LocalTensor<DT_INPUT_X> outputLocal =
                outputQueue_.DeQue<DT_INPUT_X>();
            CopyOutData(outputLocal, offset, currentLength);
            outputQueue_.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void CopyOutData(const AscendC::LocalTensor<DT_INPUT_X> &outputLocal,
        uint32_t offset, uint32_t currentLength)
    {
        const uint32_t copyBytes = currentLength * sizeof(DT_INPUT_X);
        if ((copyBytes & (DATA_BLOCK_BYTES - 1U)) == 0) {
            AscendC::DataCopy(outputGm_[offset], outputLocal, currentLength);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    AscendC::TQueBind<AscendC::TPosition::VECIN,
        AscendC::TPosition::VECOUT, BUFFER_NUM> inOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuffer2_;
    AscendC::GlobalTensor<DT_INPUT_X> inputGm_;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm_;
    uint32_t blockLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    // Host侧固定启动平台全部AIV核。先完成切分并让空闲核返回，
    // 避免小张量场景中的无效TPipe/TQue/TBuf初始化。
    const uint32_t length = tiling_data.length;
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    // 以最小搬运块对齐分核，既满足GM访存约束，也保留小张量的多核并行度。
    const uint32_t alignNum = DATA_BLOCK_BYTES / sizeof(DT_INPUT_X);
    const uint32_t average = (length + blockNum - 1U) / blockNum;
    const uint32_t blockFactor = ((average + alignNum - 1U) / alignNum) * alignNum;
    const uint32_t blockOffset = blockIdx * blockFactor;
    if (blockOffset >= length) {
        return;
    }

    uint32_t blockLength = length - blockOffset;
    if (blockLength > blockFactor) {
        blockLength = blockFactor;
    }

    AscendC::TPipe pipe;
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, blockOffset, blockLength, pipe);
    op.Process();
}
