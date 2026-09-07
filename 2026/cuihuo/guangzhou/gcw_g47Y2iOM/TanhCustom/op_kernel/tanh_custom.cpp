#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        tileLength = blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockLength * AscendC::GetBlockIdx(),
                            blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + blockLength * AscendC::GetBlockIdx(),
                            blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> expPos = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNeg = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> numerator = tmpBuf2.Get<DTYPE_X>();

        AscendC::Exp(expPos, xLocal, tileLength);
        AscendC::Muls(expNeg, xLocal, static_cast<DTYPE_X>(-1.0), tileLength);
        AscendC::Exp(expNeg, expNeg, tileLength);
        AscendC::Sub(numerator, expPos, expNeg, tileLength);
        AscendC::Add(yLocal, expPos, expNeg, tileLength);
        AscendC::Div(yLocal, numerator, yLocal, tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * tileLength], yLocal, tileLength);
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
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}

