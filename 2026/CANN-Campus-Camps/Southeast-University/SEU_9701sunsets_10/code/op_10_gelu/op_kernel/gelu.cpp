#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData& tiling)
    {
        length = tiling.length;
        blockLength = tiling.blockLength;
        tileLength = tiling.tileLength;
        mode = tiling.mode;

        uint32_t startOffset = GetBlockIdx() * blockLength;

        if (startOffset >= length) {
            currentBlockLength = 0;
            return;
        }

        currentBlockLength = length - startOffset;
        currentBlockLength = currentBlockLength > blockLength ? blockLength : currentBlockLength;

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + startOffset, currentBlockLength);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + startOffset, currentBlockLength);

        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuffer, tileLength * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process()
    {
        if (currentBlockLength == 0) {
            return;
        }

        uint32_t loopCount = (currentBlockLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t progress = i * tileLength;
            uint32_t currentNum = currentBlockLength - progress;
            currentNum = currentNum > tileLength ? tileLength : currentNum;

            CopyIn(progress, currentNum);
            Compute(currentNum);
            CopyOut(progress, currentNum);
        }
    }

private:
    __aicore__ inline uint32_t AlignNum(uint32_t num)
    {
        constexpr uint32_t alignElem = 32 / sizeof(DT_INPUT_X);
        return (num + alignElem - 1) / alignElem * alignElem;
    }

    __aicore__ inline bool IsAligned(uint32_t num)
    {
        constexpr uint32_t alignElem = 32 / sizeof(DT_INPUT_X);
        return (num % alignElem) == 0;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentNum)
    {
        LocalTensor<DT_INPUT_X> x = inputQueue.AllocTensor<DT_INPUT_X>();

        if (IsAligned(currentNum)) {
            DataCopy(x, inputGm[progress], currentNum);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = currentNum * sizeof(DT_INPUT_X);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;

            DataCopyPadExtParams<DT_INPUT_X> pp;
            pp.isPad = true;
            pp.leftPadding = 0;
            pp.rightPadding = static_cast<uint8_t>(AlignNum(currentNum) - currentNum);
            pp.paddingValue = static_cast<DT_INPUT_X>(0);

            DataCopyPad(x, inputGm[progress], cp, pp);
        }

        inputQueue.EnQue(x);
    }

    __aicore__ inline void ComputeErf(LocalTensor<DT_INPUT_X>& x,
                                      LocalTensor<DT_INPUT_X>& y,
                                      uint32_t n)
    {
        Muls(y, x, static_cast<DT_INPUT_X>(0.7071067811865476), n);
        Erf(y, y, n);
        Adds(y, y, static_cast<DT_INPUT_X>(1.0), n);
        Mul(y, x, y, n);
        Muls(y, y, static_cast<DT_INPUT_X>(0.5), n);
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_INPUT_X>& x,
                                       LocalTensor<DT_INPUT_X>& y,
                                       uint32_t n)
    {
        LocalTensor<DT_INPUT_X> x2 = tmpBuffer.Get<DT_INPUT_X>();

        // x2 = x^2
        Mul(x2, x, x, n);

        // y = (((c8*x2 + c6)*x2 + c4)*x2 + c2)*x2 + 0.5*x
        Muls(y, x2, static_cast<DT_INPUT_X>(-0.00118732821548), n);
        Adds(y, y, static_cast<DT_INPUT_X>(0.00997355701004), n);
        Mul(y, y, x2, n);
        Adds(y, y, static_cast<DT_INPUT_X>(-0.06649038006691), n);
        Mul(y, y, x2, n);
        Adds(y, y, static_cast<DT_INPUT_X>(0.39894228040143), n);
        Mul(y, y, x2, n);
        Muls(x2, x, static_cast<DT_INPUT_X>(0.5), n);
        Add(y, y, x2, n);
    }

    __aicore__ inline void Compute(uint32_t currentNum)
    {
        LocalTensor<DT_INPUT_X> x = inputQueue.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> y = outputQueue.AllocTensor<DT_INPUT_X>();

        uint32_t n = AlignNum(currentNum);

        if (mode == 0) {
            ComputeErf(x, y, n);
        } else {
            ComputePoly(x, y, n);
        }

        outputQueue.EnQue(y);
        inputQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentNum)
    {
        LocalTensor<DT_INPUT_X> y = outputQueue.DeQue<DT_INPUT_X>();

        if (IsAligned(currentNum)) {
            DataCopy(outputGm[progress], y, currentNum);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = currentNum * sizeof(DT_INPUT_X);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;

            DataCopyPad(outputGm[progress], y, cp);
        }

        outputQueue.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<DT_INPUT_X> inputGm;
    GlobalTensor<DT_INPUT_X> outputGm;

    uint32_t length = 0;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t currentBlockLength = 0;
    uint32_t mode = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}