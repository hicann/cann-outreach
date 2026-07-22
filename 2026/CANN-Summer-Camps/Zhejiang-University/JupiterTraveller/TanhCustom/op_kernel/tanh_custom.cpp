#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());

        ASSERT(blockNum != 0);
        ASSERT(tileNum != 0);

        this->blockLength = totalLength / blockNum;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        ASSERT(this->tileLength != 0);

        xGm.SetGlobalBuffer(
            (__gm__ half *)x + this->blockLength * GetBlockIdx(),
            this->blockLength);
        yGm.SetGlobalBuffer(
            (__gm__ half *)y + this->blockLength * GetBlockIdx(),
            this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));

        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();

        DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        LocalTensor<half> tmp0 = tmpBuf0.Get<half>();
        LocalTensor<half> tmp1 = tmpBuf1.Get<half>();
        LocalTensor<half> tmp2 = tmpBuf2.Get<half>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        Muls(tmp0, xLocal, static_cast<half>(-1.0f), this->tileLength);
        Exp(tmp1, xLocal, this->tileLength);
        Exp(tmp2, tmp0, this->tileLength);
        Sub(tmp0, tmp1, tmp2, this->tileLength);
        Add(tmp1, tmp1, tmp2, this->tileLength);
        Div(yLocal, tmp0, tmp1, this->tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();

        DataCopy(
            yGm[progress * this->tileLength],
            yLocal,
            this->tileLength);

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}