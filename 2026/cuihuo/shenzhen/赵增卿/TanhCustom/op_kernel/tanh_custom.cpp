#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhCustomTilingData* tilingData)
    {
        int64_t blockOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
        int64_t remaining = tilingData->totalNum - blockOffset;
        blockLength = remaining > tilingData->blockFactor ? tilingData->blockFactor : remaining;
        tileLength = tilingData->ubFactor;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int64_t loopCount = (blockLength + tileLength - 1) / tileLength;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == loopCount - 1) ? (blockLength - i * tileLength) : tileLength;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(DTYPE_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, {false, 0, 0, 0});
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> expPos = tmpBuf0.Get<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> expNeg = tmpBuf1.Get<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> denominator = tmpBuf2.Get<DTYPE_Y>();

        AscendC::Exp(expPos, xLocal, currentNum);
        AscendC::Muls(yLocal, xLocal, static_cast<DTYPE_X>(-1.0), currentNum);
        AscendC::Exp(expNeg, yLocal, currentNum);
        AscendC::Sub(yLocal, expPos, expNeg, currentNum);
        AscendC::Add(denominator, expPos, expNeg, currentNum);
        AscendC::Div(yLocal, yLocal, denominator, currentNum);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(DTYPE_Y);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh op;
    op.Init(x, y, &tilingData);
    op.Process();
}
