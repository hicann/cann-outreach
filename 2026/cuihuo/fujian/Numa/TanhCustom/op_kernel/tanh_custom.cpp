#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(half));
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
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::LocalTensor<half> tmpTensor0 = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> tmpTensor1 = tmpBuf1.Get<half>();
        AscendC::LocalTensor<half> tmpTensor2 = tmpBuf2.Get<half>();
        
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // exp(x)
        AscendC::Exp(tmpTensor0, xLocal, this->tileLength);
        // -x
        AscendC::Muls(tmpTensor1, xLocal, (half)-1.0f, this->tileLength);
        // exp(-x)
        AscendC::Exp(tmpTensor1, tmpTensor1, this->tileLength);
        // exp(x) - exp(-x)
        AscendC::Sub(tmpTensor2, tmpTensor0, tmpTensor1, this->tileLength);
        // exp(x) + exp(-x)
        AscendC::Add(tmpTensor1, tmpTensor0, tmpTensor1, this->tileLength);
        // (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, tmpTensor2, tmpTensor1, this->tileLength);
        
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}