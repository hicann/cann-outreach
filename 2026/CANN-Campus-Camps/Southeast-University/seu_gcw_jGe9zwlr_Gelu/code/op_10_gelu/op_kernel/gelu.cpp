#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;
template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length,
        uint32_t tileLength)
    {
        this->length = length;
        this->tileLength = tileLength;

        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(input_x),
            length);

        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(output),
            length);

        // 输入
        pipe.InitBuffer(
            inQueue,
            1,
            tileLength * sizeof(DT_INPUT_X));

        // 输出
        pipe.InitBuffer(
            outQueue,
            1,
            tileLength * sizeof(DT_INPUT_X));

        // x / sqrt(2)
        pipe.InitBuffer(
            tmpBuffer1,
            tileLength * sizeof(DT_INPUT_X));

        // erf(x / sqrt(2))
        pipe.InitBuffer(
            tmpBuffer2,
            tileLength * sizeof(DT_INPUT_X));

        // Erf 的 sharedTmpBuffer
        //
        // 这里先给一个保守的临时空间。
        // 如果后面编译/运行提示 Erf 临时空间不足，
        // 再根据你当前 CANN 9.0 的实际要求调整。
        pipe.InitBuffer(
            erfTmpBuffer,
            32 * 1024);
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;

        while (offset < length) {

            uint32_t remain = length - offset;

            uint32_t curLength =
                (remain < tileLength)
                    ? remain
                    : tileLength;

            // 按32字节向上对齐
            uint32_t elementBytes =
                sizeof(DT_INPUT_X);

            uint32_t alignedBytes =
                ((curLength * elementBytes + 31) / 32) * 32;

            uint32_t alignedLength =
                alignedBytes / elementBytes;

            CopyIn(
                offset,
                curLength,
                alignedLength);

            Compute(alignedLength);

            CopyOut(
                offset,
                curLength);

            offset += curLength;
        }
    }

private:

    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t curLength,
        uint32_t alignedLength)
    {
        LocalTensor<DT_INPUT_X> inputLocal =
            inQueue.AllocTensor<DT_INPUT_X>();

        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        copyParams.blockLen =
            curLength * sizeof(DT_INPUT_X);

        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_INPUT_X> padParams;

        padParams.isPad = true;
        padParams.leftPadding = 0;

        padParams.rightPadding =
            alignedLength - curLength;

        padParams.paddingValue = 0;

        DataCopyPad(
            inputLocal,
            inputGm[offset],
            copyParams,
            padParams);

        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(
        uint32_t calCount)
    {
        LocalTensor<DT_INPUT_X> inputLocal =
            inQueue.DeQue<DT_INPUT_X>();

        LocalTensor<DT_INPUT_X> tmp1 =
            tmpBuffer1.Get<DT_INPUT_X>();

        LocalTensor<DT_INPUT_X> tmp2 =
            tmpBuffer2.Get<DT_INPUT_X>();

        LocalTensor<uint8_t> erfTmp =
            erfTmpBuffer.Get<uint8_t>();

        LocalTensor<DT_INPUT_X> outputLocal =
            outQueue.AllocTensor<DT_INPUT_X>();

        // ========================================
        // GELU(x)
        //
        // x * 0.5 * (1 + erf(x / sqrt(2)))
        // ========================================

        // 1. x / sqrt(2)
        constexpr float INV_SQRT_2 =
            0.7071067811865475f;

        Muls(
            tmp1,
            inputLocal,
            static_cast<DT_INPUT_X>(INV_SQRT_2),
            calCount);

        // 2. erf(x / sqrt(2))
        //
        // 使用 sharedTmpBuffer 版本，
        // 不再依赖 Host 的 GetErfMaxMinTmpSize。
        Erf<DT_INPUT_X, false>(
            tmp2,
            tmp1,
            erfTmp,
            calCount);

        // 3. 1 + erf(...)
        Adds(
            tmp1,
            tmp2,
            static_cast<DT_INPUT_X>(1.0f),
            calCount);

        // 4. 0.5 * (...)
        Muls(
            tmp1,
            tmp1,
            static_cast<DT_INPUT_X>(0.5f),
            calCount);

        // 5. x * (...)
        Mul(
            outputLocal,
            inputLocal,
            tmp1,
            calCount);

        outQueue.EnQue(outputLocal);

        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t curLength)
    {
        LocalTensor<DT_INPUT_X> outputLocal =
            outQueue.DeQue<DT_INPUT_X>();

        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        copyParams.blockLen =
            curLength * sizeof(DT_INPUT_X);

        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPad(
            outputGm[offset],
            outputLocal,
            copyParams);

        outQueue.FreeTensor(outputLocal);
    }

private:

    GlobalTensor<DT_INPUT_X> inputGm;
    GlobalTensor<DT_INPUT_X> outputGm;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    TBuf<TPosition::VECCALC> tmpBuffer1;
    TBuf<TPosition::VECCALC> tmpBuffer2;

    TBuf<TPosition::VECCALC> erfTmpBuffer;

    uint32_t length = 0;
    uint32_t tileLength = 0;
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
        tiling_data,
        tiling);

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tiling_data.length,
        tiling_data.tileLength);

    op.Process();
}