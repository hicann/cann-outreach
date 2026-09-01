#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + AscendC::GetBlockIdx() * totalLength, totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + AscendC::GetBlockIdx() * totalLength, totalLength);

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
        AscendC::LocalTensor<half> expPos = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> expNeg = tmpBuf1.Get<half>();
        AscendC::LocalTensor<half> tmp = tmpBuf2.Get<half>();

        // exp(x)
        AscendC::Exp(expPos, xLocal, this->tileLength);
        // -x
        AscendC::Muls(tmp, xLocal, (half)(-1.0), this->tileLength);
        // exp(-x)
        AscendC::Exp(expNeg, tmp, this->tileLength);
        // exp(x) - exp(-x)
        AscendC::Sub(yLocal, expPos, expNeg, this->tileLength);
        // exp(x) + exp(-x)
        AscendC::Add(tmp, expPos, expNeg, this->tileLength);
        // tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, yLocal, tmp, this->tileLength);

        outQueueY.EnQue(yLocal);
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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    AscendC::GlobalTensor<uint32_t> tilingGm;
    tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling, 2);
    uint32_t totalLength = tilingGm.GetValue(0);
    uint32_t tileNum = tilingGm.GetValue(1);

    KernelTanh op;
    op.Init(x, y, totalLength, tileNum);
    op.Process();
}