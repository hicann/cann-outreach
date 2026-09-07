#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu()
        : coreLength(0U),
          tileLength(0U),
          globalOffset(0U)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR inputX,
        GM_ADDR output,
        const GeluTilingData &tilingData)
    {
        constexpr uint32_t BLOCK_BYTES = 32U;
        constexpr uint32_t ELEMENTS_PER_BLOCK =
            BLOCK_BYTES / sizeof(T);

        const uint32_t blockIdx =
            AscendC::GetBlockIdx();

        const uint32_t currentBlockCount =
            tilingData.baseBlockCount +
            ((blockIdx < tilingData.extraBlockCores)
                 ? 1U
                 : 0U);

        const uint32_t precedingExtraBlocks =
            (blockIdx < tilingData.extraBlockCores)
                ? blockIdx
                : tilingData.extraBlockCores;

        const uint32_t blockOffset =
            blockIdx * tilingData.baseBlockCount +
            precedingExtraBlocks;

        globalOffset =
            blockOffset * ELEMENTS_PER_BLOCK;

        coreLength =
            currentBlockCount * ELEMENTS_PER_BLOCK;

        if (blockIdx + 1U == tilingData.coreNum) {
            coreLength += tilingData.tailElements;
        }

        tileLength = tilingData.tileLength;

        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(inputX) +
                globalOffset,
            coreLength);

        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(output) +
                globalOffset,
            coreLength);

        const uint32_t tileBytes =
            tileLength * sizeof(T);

        pipe.InitBuffer(
            inputQueue,
            1,
            tileBytes);

        pipe.InitBuffer(
            outputQueue,
            1,
            tileBytes);

        pipe.InitBuffer(
            scaledBuffer,
            tileBytes);

        /*
         * float32快速路径不调用Erf，
         * 因此不分配Erf临时空间。
         */
        if constexpr (!AscendC::IsSameType<T, float>::value) {
            pipe.InitBuffer(
                erfTmpBuffer,
                tilingData.tmpBufferSize);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0U) {
            return;
        }

        const uint32_t loopCount =
            (coreLength + tileLength - 1U) /
            tileLength;

        for (uint32_t loopIdx = 0U;
             loopIdx < loopCount;
             ++loopIdx) {
            const uint32_t offset =
                loopIdx * tileLength;

            const uint32_t remaining =
                coreLength - offset;

            const uint32_t currentLength =
                (remaining < tileLength)
                    ? remaining
                    : tileLength;

            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(
        const uint32_t offset,
        const uint32_t currentLength)
    {
        constexpr uint32_t BLOCK_BYTES = 32U;

        AscendC::LocalTensor<T> inputLocal =
            inputQueue.AllocTensor<T>();

        const uint32_t copyBytes =
            currentLength * sizeof(T);

        if ((copyBytes % BLOCK_BYTES) == 0U) {
            AscendC::DataCopy(
                inputLocal,
                inputGm[offset],
                currentLength);
        } else {
            AscendC::DataCopyPad(
                inputLocal,
                inputGm[offset],
                {
                    1,
                    static_cast<uint16_t>(copyBytes),
                    0,
                    0
                },
                {
                    true,
                    0,
                    0,
                    0
                });
        }

        inputQueue.EnQue<T>(inputLocal);
    }

    __aicore__ inline void Compute(
        const uint32_t currentLength)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue.DeQue<T>();

        AscendC::LocalTensor<T> outputLocal =
            outputQueue.AllocTensor<T>();

        AscendC::LocalTensor<T> scaledLocal =
            scaledBuffer.Get<T>();

        if constexpr (AscendC::IsSameType<T, float>::value) {
            /*
             * 高精度快速近似：
             *
             * GELU(x) ≈ x / (1 + exp(-P(x)))
             *
             * P(x) = x * (
             *       1.59568629
             *     + 0.0728700826 * x^2
             *     - 0.000204218720 * x^4
             *     - 0.0000715244894 * x^6
             *     + 0.00000294458706 * x^8
             * )
             */

            // scaledLocal = x^2
            AscendC::Mul(
                scaledLocal,
                inputLocal,
                inputLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            /*
             * Horner形式计算-P(x)：
             *
             * q = -c8*x^2 - c6
             * q = q*x^2 - c4
             * q = q*x^2 - c2
             * q = q*x^2 - c0
             * -P(x) = q*x
             */

            AscendC::Muls(
                outputLocal,
                scaledLocal,
                -2.94458706e-6f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Adds(
                outputLocal,
                outputLocal,
                7.15244894e-5f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(
                outputLocal,
                outputLocal,
                scaledLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Adds(
                outputLocal,
                outputLocal,
                2.04218720e-4f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(
                outputLocal,
                outputLocal,
                scaledLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Adds(
                outputLocal,
                outputLocal,
                -7.28700826e-2f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(
                outputLocal,
                outputLocal,
                scaledLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Adds(
                outputLocal,
                outputLocal,
                -1.59568629f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            // outputLocal = -P(x)
            AscendC::Mul(
                outputLocal,
                outputLocal,
                inputLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            // scaledLocal = exp(-P(x))
            AscendC::Exp(
                scaledLocal,
                outputLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            // scaledLocal = 1 + exp(-P(x))
            AscendC::Adds(
                scaledLocal,
                scaledLocal,
                1.0f,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            // outputLocal = x / (1 + exp(-P(x)))
            AscendC::Div(
                outputLocal,
                inputLocal,
                scaledLocal,
                currentLength);
        } else {
            /*
             * float16继续使用Erf路径。
             * 避免小系数在half下精度不足或发生下溢。
             */
            AscendC::LocalTensor<uint8_t> tmpLocal =
                erfTmpBuffer.Get<uint8_t>();

            const T invSqrt2 =
                static_cast<T>(0.7071067811865475f);

            const T halfValue =
                static_cast<T>(0.5f);

            // scaledLocal = x / sqrt(2)
            AscendC::Muls(
                scaledLocal,
                inputLocal,
                invSqrt2,
                currentLength);

            // outputLocal = x / 2
            AscendC::Muls(
                outputLocal,
                inputLocal,
                halfValue,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            // inputLocal = erf(x / sqrt(2))
            AscendC::Erf<T, false>(
                inputLocal,
                scaledLocal,
                tmpLocal,
                currentLength);

            AscendC::PipeBarrier<PIPE_V>();

            /*
             * outputLocal =
             * x/2 + (x/2)*erf(x/sqrt(2))
             */
            AscendC::MulAddDst(
                outputLocal,
                outputLocal,
                inputLocal,
                currentLength);
        }

        outputQueue.EnQue<T>(outputLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        const uint32_t offset,
        const uint32_t currentLength)
    {
        constexpr uint32_t BLOCK_BYTES = 32U;

        AscendC::LocalTensor<T> outputLocal =
            outputQueue.DeQue<T>();

        const uint32_t copyBytes =
            currentLength * sizeof(T);

        if ((copyBytes % BLOCK_BYTES) == 0U) {
            AscendC::DataCopy(
                outputGm[offset],
                outputLocal,
                currentLength);
        } else {
            AscendC::DataCopyPad(
                outputGm[offset],
                outputLocal,
                {
                    1,
                    static_cast<uint16_t>(copyBytes),
                    0,
                    0
                });
        }

        outputQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1>
        inputQueue;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1>
        outputQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC>
        scaledBuffer;

    AscendC::TBuf<AscendC::TPosition::VECCALC>
        erfTmpBuffer;

    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> outputGm;

    uint32_t coreLength;
    uint32_t tileLength;
    uint32_t globalOffset;
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

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tilingData);

    op.Process();
}