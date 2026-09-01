#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_Y));
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
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> tmpExpPos = tmpBuf0.Get<DTYPE_Y>();   // 存 e^x
        LocalTensor<DTYPE_Y> tmpExpNeg = tmpBuf1.Get<DTYPE_Y>();   // 存 e^{-x}
        LocalTensor<DTYPE_Y> tmpDenom = tmpBuf2.Get<DTYPE_Y>();    // 存分母 e^x + e^{-x}

        // 1. 计算 e^x
        Exp(tmpExpPos, xLocal, this->tileLength);

        // 2. 计算 -x，然后 e^{-x}
        half negOne = -1.0;
        Muls(tmpExpNeg, xLocal, negOne, this->tileLength);  // tmpExpNeg = -x
        Exp(tmpExpNeg, tmpExpNeg, this->tileLength);        // tmpExpNeg = e^{-x}

        // 3. tanh = (e^x - e^{-x}) / (e^x + e^{-x})
        // yLocal = e^x - e^{-x}
        Sub(yLocal, tmpExpPos, tmpExpNeg, this->tileLength);
        // tmpDenom = e^x + e^{-x}
        Add(tmpDenom, tmpExpPos, tmpExpNeg, this->tileLength);
        // yLocal = yLocal / tmpDenom = tanh
        Div(yLocal, yLocal, tmpDenom, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * tileLength], yLocal, tileLength);
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
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();

}