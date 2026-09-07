/*!
 * \file gelu.cpp
 * \brief GELU算子Kernel实现
 */

#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {

constexpr int32_t QUEUE_DEPTH = 1;
constexpr uint8_t MULTI_TILE_BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

// 在x∈[-1,1]范围内，对下面的形式进行误差优化：
//
// GELU(x) ≈ 0.5*x + u*(A0 + A1*u + A2*u*u)
// u = x*x
//
// 相比上一版三次P(u)，本版减少一层Mul和Adds。
constexpr float GELU_A0 = 0.398765504360199f;
constexpr float GELU_A1 = -0.0653865858912468f;
constexpr float GELU_A2 = 0.007974408566951752f;

} // namespace

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR inputX,
        GM_ADDR output,
        const GeluTilingData& tilingData)
    {
        const uint32_t blockOffset =
            AscendC::GetBlockIdx() *
            tilingData.blockLength;

        uint32_t remainingLength = 0;

        if (blockOffset < tilingData.totalLength) {
            remainingLength =
                tilingData.totalLength - blockOffset;
        }

        blockLength_ =
            remainingLength < tilingData.blockLength
                ? remainingLength
                : tilingData.blockLength;

        tileLength_ = tilingData.tileLength;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(inputX) +
                blockOffset,
            blockLength_);

        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output) +
                blockOffset,
            blockLength_);

        // 单Tile使用单Buffer，减少小张量资源管理开销；
        // 多Tile使用Double Buffer。
        const uint8_t bufferNum =
            blockLength_ <= tileLength_
                ? 1
                : MULTI_TILE_BUFFER_NUM;

        pipe_.InitBuffer(
            inputQueue_,
            bufferNum,
            tileLength_ * sizeof(T));

        pipe_.InitBuffer(
            outputQueue_,
            bufferNum,
            tileLength_ * sizeof(T));

        // 保存u=x*x
        pipe_.InitBuffer(
            temporaryBuffer_,
            tileLength_ * sizeof(float));

        // FP16计算时使用FP32中间数据，保证精度。
        if constexpr (
            AscendC::IsSameType<T, half>::value) {
            pipe_.InitBuffer(
                inputFloatBuffer_,
                tileLength_ * sizeof(float));

            pipe_.InitBuffer(
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
            const uint32_t remainingLength =
                blockLength_ - offset;

            const uint32_t currentLength =
                remainingLength < tileLength_
                    ? remainingLength
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
        uint32_t currentLength)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.AllocTensor<T>();

        const uint32_t currentBytes =
            currentLength * sizeof(T);

        if ((currentBytes % ALIGN_BYTES) == 0) {
            AscendC::DataCopy(
                inputLocal,
                inputGm_[offset],
                currentLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                currentBytes,
                0,
                0,
                0
            };

            const uint32_t alignedBytes =
                ((currentBytes + ALIGN_BYTES - 1) /
                 ALIGN_BYTES) *
                ALIGN_BYTES;

            const uint8_t rightPadding =
                static_cast<uint8_t>(
                    (alignedBytes - currentBytes) /
                    sizeof(T));

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

        inputQueue_.EnQue(inputLocal);
    }

    /*
     * 优化后的计算过程：
     *
     * result = 0.5*x
     * u      = x*x
     * p      = A2*u + A1
     * p      = p*u + A0
     * result = result + u*p
     *
     * 最后一条使用MulAddDst合并乘法与加法。
     *
     * input在计算出u后不再需要，因此复用input保存p，
     * 避免再申请一个临时Buffer。
     */
    __aicore__ inline void ComputePolynomial(
        AscendC::LocalTensor<float> input,
        const AscendC::LocalTensor<float>& result,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<float> temporary =
            temporaryBuffer_.Get<float>();

        // result = 0.5*x
        AscendC::Muls(
            result,
            input,
            0.5f,
            currentLength);

        // temporary = u = x*x
        AscendC::Mul(
            temporary,
            input,
            input,
            currentLength);

        // 原输入x已经不再需要，复用input保存多项式p。
        //
        // input = A2*u
        AscendC::Muls(
            input,
            temporary,
            GELU_A2,
            currentLength);

        // input = A2*u + A1
        AscendC::Adds(
            input,
            input,
            GELU_A1,
            currentLength);

        // input = (A2*u + A1)*u
        AscendC::Mul(
            input,
            input,
            temporary,
            currentLength);

        // input = (A2*u + A1)*u + A0
        AscendC::Adds(
            input,
            input,
            GELU_A0,
            currentLength);

        // result = result + temporary*input
        //
        // 即：
        // result = 0.5*x + u*P(u)
        AscendC::MulAddDst(
            result,
            temporary,
            input,
            currentLength);
    }

    __aicore__ inline void Compute(
        uint32_t currentLength)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.DeQue<T>();

        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.AllocTensor<T>();

        if constexpr (
            AscendC::IsSameType<T, half>::value) {
            AscendC::LocalTensor<float> inputFloat =
                inputFloatBuffer_.Get<float>();

            AscendC::LocalTensor<float> resultFloat =
                resultFloatBuffer_.Get<float>();

            AscendC::Cast(
                inputFloat,
                inputLocal,
                AscendC::RoundMode::CAST_NONE,
                currentLength);

            ComputePolynomial(
                inputFloat,
                resultFloat,
                currentLength);

            AscendC::Cast(
                outputLocal,
                resultFloat,
                AscendC::RoundMode::CAST_RINT,
                currentLength);
        } else {
            ComputePolynomial(
                inputLocal,
                outputLocal,
                currentLength);
        }

        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.DeQue<T>();

        const uint32_t currentBytes =
            currentLength * sizeof(T);

        if ((currentBytes % ALIGN_BYTES) == 0) {
            AscendC::DataCopy(
                outputGm_[offset],
                outputLocal,
                currentLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                currentBytes,
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
    AscendC::TPipe pipe_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        QUEUE_DEPTH> inputQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        QUEUE_DEPTH> outputQueue_;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC>
        temporaryBuffer_;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC>
        inputFloatBuffer_;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC>
        resultFloatBuffer_;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR inputX,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tilingData,
        tiling);

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        inputX,
        output,
        tilingData);

    op.Process();

    (void)workspace;
}