#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength=totalLength/AscendC::GetBlockNum();
        this->tileNum=tileNum;
        this->tileLength=this->blockLength/tileNum/BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x+this->blockLength*AscendC::GetBlockIdx(),this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y+this->blockLength*AscendC::GetBlockIdx(),this->blockLength);
        pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0,this->tileLength*sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1,this->tileLength*sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2,this->tileLength*sizeof(DTYPE_X));
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
        AscendC::LocalTensor<DTYPE_X> xLocal=inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
        inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal=inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal=outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> expX=tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNegX=tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> temp=tmpBuf2.Get<DTYPE_X>();
        AscendC::Muls(temp,xLocal,(DTYPE_X)-1.0,this->tileLength);
        AscendC::Exp(expX,xLocal,this->tileLength);
        AscendC::Exp(expNegX,temp,this->tileLength);
        AscendC::Sub(yLocal,expX,expNegX,this->tileLength);
        AscendC::Add(temp,expX,expNegX,this->tileLength);
        AscendC::Div(yLocal,yLocal,temp,this->tileLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal=outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress*this->tileLength],yLocal,this->tileLength);
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
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x,y,tilingData.totalLength,tilingData.tileNum);
    op.Process();

}