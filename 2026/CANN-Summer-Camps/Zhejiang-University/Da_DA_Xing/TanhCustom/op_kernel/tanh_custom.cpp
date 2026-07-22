#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(expXBuf, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(expNegXBuf, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(numeratorBuf, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(denominatorBuf, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO: 考生自行补齐
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> numeratorLocal = numeratorBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> denominatorLocal = denominatorBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expXLocal = expXBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNegXLocal = expNegXBuf.Get<DTYPE_X>();

        AscendC::Muls(numeratorLocal, xLocal, static_cast<DTYPE_X>(-1.0f), this->tileLength);//-x
        AscendC::Exp(expXLocal, xLocal, this->tileLength);//exp(x)
        AscendC::Exp(expNegXLocal, numeratorLocal, this->tileLength);//exp(-x)
        AscendC::Sub(numeratorLocal, expXLocal, expNegXLocal, this->tileLength);//sub
        AscendC::Add(denominatorLocal, expXLocal, expNegXLocal, this->tileLength);//add
        AscendC::Div(yLocal, numeratorLocal, denominatorLocal, this->tileLength);//div

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);


    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> expXBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> expNegXBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> numeratorBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> denominatorBuf;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: 考生自行补齐
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void tanh_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *workspace, uint8_t *tiling)
{
    tanh_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif