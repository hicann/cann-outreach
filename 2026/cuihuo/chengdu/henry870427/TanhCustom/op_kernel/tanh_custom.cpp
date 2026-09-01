#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        // 注意向上对齐
        this->tileLength =  (totalLength + tileNum * BUFFER_NUM - 1) / (tileNum * BUFFER_NUM);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);

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
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        // 公式tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // X
        // Y
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNX = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();

        // tmp 计算得到 -x
        AscendC::Muls(tmp, xLocal, (DTYPE_X)(-1.0f), tileLength);
        // expX = exp(x)
        AscendC::Exp(expX, xLocal, tileLength);
        // expNX = exp(-x),tmp已经是-x了
        AscendC::Exp(expNX, tmp, tileLength);
        // exp(x) - exp(-x)，结果暂存tmp
        AscendC::Sub(tmp, expX, expNX, tileLength);
         // exp(x) + exp(-x)，结果暂存expX
        AscendC::Add(expX, expX, expNX, tileLength);
        // 除法，结果直接放到yLocal
        AscendC::Div(yLocal, tmp, expX, tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
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
    // TODO: 考生自行补齐
    KernelTanh op;
    op.Init(x, y, tilingData.blockLength, tilingData.tileNum);
    op.Process();
}
