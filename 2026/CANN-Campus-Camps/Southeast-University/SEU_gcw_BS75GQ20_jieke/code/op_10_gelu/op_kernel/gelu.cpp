#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {

constexpr int32_t QUEUE_DEPTH = 1;
constexpr int32_t MULTI_TILE_BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

// 1 / sqrt(2 * pi)
constexpr float INV_SQRT_TWO_PI =
    0.39894228040143267794f;

// 多项式系数
constexpr float GELU_C1 =
    -0.16666666666666666667f;  // -1 / 6

constexpr float GELU_C2 =
    0.025f;                    // 1 / 40

constexpr float GELU_C3 =
    -0.00297619047619047619f;  // -1 / 336

constexpr float GELU_C4 =
    0.00028935185185185185f;   // 1 / 3456

} // namespace

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const GeluTilingData& tilingData,
        AscendC::TPipe* pipe)
    {
        const uint64_t blockOffset =
            static_cast<uint64_t>(
                AscendC::GetBlockIdx())
            * tilingData.blockLength;

        const uint64_t totalLength =
            tilingData.totalLength;

        if (blockOffset >= totalLength) {
            blockLength_ = 0;
            return;
        }

        const uint64_t remaining =
            totalLength - blockOffset;

        blockLength_ =
            remaining < tilingData.blockLength
                ? static_cast<uint32_t>(remaining)
                : tilingData.blockLength;

        tileLength_ =
            tilingData.tileLength;

        inputGm_.SetGlobalBuffer(
            (__gm__ T*)input_x + blockOffset,
            blockLength_);

        outputGm_.SetGlobalBuffer(
            (__gm__ T*)output + blockOffset,
            blockLength_);

        // 单Tile不需要双缓冲。
        // 多Tile继续保留双缓冲。
        const int32_t bufferNum =
            blockLength_ <= tileLength_
                ? 1
                : MULTI_TILE_BUFFER_NUM;

        pipe->InitBuffer(
            inputQueue_,
            bufferNum,
            tileLength_ * sizeof(T));

        pipe->InitBuffer(
            outputQueue_,
            bufferNum,
            tileLength_ * sizeof(T));

        // 保存u = x²
        pipe->InitBuffer(
            squareBuffer_,
            tileLength_ * sizeof(float));

        if constexpr (sizeof(T) == sizeof(half)) {
            // FP16转为FP32计算
            pipe->InitBuffer(
                inputFloatBuffer_,
                tileLength_ * sizeof(float));

            // 保存FP32计算结果
            pipe->InitBuffer(
                resultFloatBuffer_,
                tileLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }

        // 单Tile快速路径
        if (blockLength_ <= tileLength_) {
            CopyIn(0, blockLength_);
            Compute(blockLength_);
            CopyOut(0, blockLength_);
            return;
        }

        uint32_t offset = 0;

        while (offset < blockLength_) {
            const uint32_t remaining =
                blockLength_ - offset;

            const uint32_t currentLength =
                remaining < tileLength_
                    ? remaining
                    : tileLength_;

            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);

            offset += currentLength;
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.AllocTensor<T>();

        const uint32_t bytes =
            count * sizeof(T);

        if (bytes % ALIGN_BYTES == 0) {
            AscendC::DataCopy(
                inputLocal,
                inputGm_[offset],
                count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                bytes,
                0,
                0,
                0
            };

            constexpr uint32_t alignElements =
                ALIGN_BYTES / sizeof(T);

            const uint32_t remainder =
                count % alignElements;

            const uint8_t rightPadding =
                remainder == 0
                    ? 0
                    : static_cast<uint8_t>(
                        alignElements - remainder);

            AscendC::DataCopyPadExtParams<T> padParams{
                true,
                0,
                rightPadding,
                static_cast<T>(0)
            };

            AscendC::DataCopyPad(
                inputLocal,
                inputGm_[offset],
                copyParams,
                padParams);
        }

        inputQueue_.EnQue<T>(inputLocal);
    }

    // ------------------------------------------------------------
    // FP32多项式计算
    //
    // u = x²
    //
    // p = 1
    //     - u / 6
    //     + u² / 40
    //     - u³ / 336
    //     + u⁴ / 3456
    //
    // y = 0.5x + x² * p / sqrt(2pi)
    //
    // Horner形式减少中间Tensor和指令数量。
    // ------------------------------------------------------------
    __aicore__ inline void ComputePolynomial(
        const AscendC::LocalTensor<float>& input,
        const AscendC::LocalTensor<float>& square,
        const AscendC::LocalTensor<float>& result,
        uint32_t count)
    {
        // u = x²
        AscendC::Mul(
            square,
            input,
            input,
            count);

        // result = C4 * u + C3
        AscendC::Muls(
            result,
            square,
            GELU_C4,
            count);

        AscendC::Adds(
            result,
            result,
            GELU_C3,
            count);

        // result = result * u + C2
        AscendC::Mul(
            result,
            result,
            square,
            count);

        AscendC::Adds(
            result,
            result,
            GELU_C2,
            count);

        // result = result * u + C1
        AscendC::Mul(
            result,
            result,
            square,
            count);

        AscendC::Adds(
            result,
            result,
            GELU_C1,
            count);

        // result = result * u + 1
        AscendC::Mul(
            result,
            result,
            square,
            count);

        AscendC::Adds(
            result,
            result,
            1.0f,
            count);

        // result = u * result
        AscendC::Mul(
            result,
            result,
            square,
            count);

        // result = u * result / sqrt(2pi)
        AscendC::Muls(
            result,
            result,
            INV_SQRT_TWO_PI,
            count);

        // square缓冲区已经不再需要保存x²，
        // 复用它保存0.5x。
        AscendC::Muls(
            square,
            input,
            0.5f,
            count);

        // result = 0.5x + u*p/sqrt(2pi)
        AscendC::Add(
            result,
            result,
            square,
            count);
    }

    __aicore__ inline void Compute(
        uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.DeQue<T>();

        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.AllocTensor<T>();

        AscendC::LocalTensor<float> squareLocal =
            squareBuffer_.Get<float>();

        if constexpr (sizeof(T) == sizeof(half)) {
            AscendC::LocalTensor<float> inputFloat =
                inputFloatBuffer_.Get<float>();

            AscendC::LocalTensor<float> resultFloat =
                resultFloatBuffer_.Get<float>();

            // FP16 -> FP32
            AscendC::Cast(
                inputFloat,
                inputLocal,
                AscendC::RoundMode::CAST_NONE,
                count);

            ComputePolynomial(
                inputFloat,
                squareLocal,
                resultFloat,
                count);

            // FP32 -> FP16
            AscendC::Cast(
                outputLocal,
                resultFloat,
                AscendC::RoundMode::CAST_RINT,
                count);
        } else {
            // FP32直接使用输出Queue缓冲区保存结果，
            // 不再申请额外结果Tensor。
            ComputePolynomial(
                inputLocal,
                squareLocal,
                outputLocal,
                count);
        }

        outputQueue_.EnQue<T>(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t count)
    {
        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.DeQue<T>();

        const uint32_t bytes =
            count * sizeof(T);

        if (bytes % ALIGN_BYTES == 0) {
            AscendC::DataCopy(
                outputGm_[offset],
                outputLocal,
                count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                bytes,
                0,
                0,
                0
            };

            AscendC::DataCopyPad(
                outputGm_[offset],
                outputLocal,
                copyParams);
        }

        outputQueue_.FreeTensor(outputLocal);
    }

private:
    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        QUEUE_DEPTH> inputQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        QUEUE_DEPTH> outputQueue_;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> squareBuffer_;

    // 以下两个Buffer仅在FP16分支中初始化和使用。
    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> inputFloatBuffer_;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> resultFloatBuffer_;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tilingData,
        tiling);

    AscendC::TPipe pipe;
    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tilingData,
        &pipe);

    op.Process();

    (void)workspace;
}