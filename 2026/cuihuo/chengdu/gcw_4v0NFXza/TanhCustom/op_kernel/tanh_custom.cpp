/**
 * @file tanh_custom.cpp
 *
 * TanhCustom kernel, structure aligned with AddCustom sample.
 * Compute: tanh(x) = (exp(x) - exp(-x)) * reciprocal(exp(x) + exp(-x))
 */
#include "kernel_operator.h"

// 输入输出类型为 Float16，对应作业要求的 half
#define DTYPE_X half
#define DTYPE_Y half

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> t0 = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> t1 = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> t2 = tmpBuf2.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // t0 = exp(x)
        AscendC::Exp(t0, xLocal, this->tileLength);
        // t1 = -x
        AscendC::Muls(t1, xLocal, (DTYPE_X)-1.0, this->tileLength);
        // t1 = exp(-x)
        AscendC::Exp(t1, t1, this->tileLength);
        // t2 = exp(x) - exp(-x)
        AscendC::Sub(t2, t0, t1, this->tileLength);
        // t1 = exp(x) + exp(-x)
        AscendC::Add(t1, t0, t1, this->tileLength);
        // y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, t2, t1, this->tileLength);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelTanh op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void tanh_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y,
                    uint8_t *workspace, uint8_t *tiling)
{
    tanh_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
