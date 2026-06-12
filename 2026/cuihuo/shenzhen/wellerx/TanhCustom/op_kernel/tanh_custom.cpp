#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = blockLength / tileNum;
        this->tileLength = this->tileLength / 16 * 16;

        if (this->tileLength == 0) {
            this->tileLength = 16;
            this->tileNum = 1;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
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

        AscendC::Exp(expPos, xLocal, this->tileLength);

        AscendC::Muls(expNeg, xLocal, static_cast<half>(-1), this->tileLength);
        AscendC::Exp(expNeg, expNeg, this->tileLength);

        AscendC::Sub(yLocal, expPos, expNeg, this->tileLength);

        AscendC::Add(expPos, expPos, expNeg, this->tileLength);

        AscendC::Div(yLocal, yLocal, expPos, this->tileLength);

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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.totalLength;
    uint32_t tileNum = tilingData.tileNum;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockDim = AscendC::GetBlockNum();
    uint32_t blockLength = totalLength / blockDim;
    uint32_t blockOffset = blockIdx * blockLength;

    if (blockIdx == blockDim - 1) {
        blockLength = totalLength - blockOffset;
    }

    if (blockLength > 0) {
        KernelTanh op;
        op.Init(x + blockOffset * sizeof(half),
                y + blockOffset * sizeof(half),
                blockLength, tileNum);
        op.Process();
    }
}
