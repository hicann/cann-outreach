#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);

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
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<half> expPosTensor = tmpBuf0.Get<half>();
        LocalTensor<half> expNegTensor = tmpBuf1.Get<half>();
        LocalTensor<half> sumTensor = tmpBuf2.Get<half>();

        // exp(x)
        Exp(expPosTensor, xLocal, this->tileLength);
        // Build -x by dividing x by -1.0, then compute exp(-x).
        half negOne(-1.0);
        Duplicate<half>(expNegTensor, negOne, this->tileLength);
        Div(expNegTensor, xLocal, expNegTensor, this->tileLength);
        Exp(expNegTensor, expNegTensor, this->tileLength);

        // numerator = exp(x) - exp(-x)
        yLocal = expPosTensor - expNegTensor;
        // denominator = exp(x) + exp(-x)
        sumTensor = expPosTensor + expNegTensor;
        Div(yLocal, yLocal, sumTensor, this->tileLength);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
