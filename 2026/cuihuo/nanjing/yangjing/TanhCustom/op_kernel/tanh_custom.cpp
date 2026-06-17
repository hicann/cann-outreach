#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->tileNum = tileNum;
        this->tileLength = totalLength / (AscendC::GetBlockNum() * tileNum);
        this->blockLength = this->tileLength * sizeof(DTYPE_X);
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength);
        pipe.InitBuffer(tmpBuf0, this->blockLength);
        pipe.InitBuffer(tmpBuf1, this->blockLength);
        pipe.InitBuffer(tmpBuf2, this->blockLength);
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        uint32_t offset = AscendC::GetBlockIdx() * this->tileNum * this->tileLength + progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        AscendC::LocalTensor<half> tmp0 = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> tmp1 = tmpBuf1.Get<half>();
        AscendC::LocalTensor<half> tmp2 = tmpBuf2.Get<half>();
        
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Exp(tmp0, xLocal, this->tileLength);           // exp(x)
        AscendC::Muls(tmp1, xLocal, static_cast<half>(-1.0), this->tileLength);  // -x
        AscendC::Exp(tmp1, tmp1, this->tileLength);              // exp(-x)
        AscendC::Sub(tmp2, tmp0, tmp1, this->tileLength);       // exp(x) - exp(-x)
        AscendC::Add(tmp0, tmp0, tmp1, this->tileLength);       // exp(x) + exp(-x)
        AscendC::Div(yLocal, tmp2, tmp0, this->tileLength);     // tanh
        
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t offset = AscendC::GetBlockIdx() * this->tileNum * this->tileLength + progress * this->tileLength;
        AscendC::DataCopy(yGm[offset], yLocal, this->tileLength);
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
    uint32_t totalLength = tilingData.totalLength;
    uint32_t tileNum = tilingData.tileNum;
    op.Init(x, y, totalLength, tileNum);
    op.Process();
}
