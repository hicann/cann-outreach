#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        ASSERT(
            AscendC::GetBlockNum() != 0 &&
            "block dim can not be zero!");

        ASSERT(
            tileNum != 0 &&
            "tile num can not be zero!");

        this->blockLength =
            totalLength /
            AscendC::GetBlockNum();

        this->tileNum = tileNum;

        this->tileLength =
            this->blockLength /
            this->tileNum /
            BUFFER_NUM;

        uint32_t offset =
            this->blockLength *
            AscendC::GetBlockIdx();

        this->xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X*)x + offset,
            this->blockLength);

        this->yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y*)y + offset,
            this->blockLength);

        this->pipe.InitBuffer(
            this->inQueueX,
            BUFFER_NUM,
            this->tileLength *
                sizeof(DTYPE_X));

        this->pipe.InitBuffer(
            this->outQueueY,
            BUFFER_NUM,
            this->tileLength *
                sizeof(DTYPE_Y));

        this->pipe.InitBuffer(
            this->tmpBuf0,
            this->tileLength *
                sizeof(DTYPE_X));

        this->pipe.InitBuffer(
            this->tmpBuf1,
            this->tileLength *
                sizeof(DTYPE_X));

        this->pipe.InitBuffer(
            this->tmpBuf2,
            this->tileLength *
                sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0;
             i < loopCount;
             i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
            this->inQueueX
                .AllocTensor<DTYPE_X>();

        AscendC::DataCopy(
            xLocal,
            this->xGm[
                progress *
                this->tileLength],
            this->tileLength);

        this->inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(
        int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
            this->inQueueX
                .DeQue<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_Y> yLocal =
            this->outQueueY
                .AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> tmpLocal0 =
            this->tmpBuf0.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmpLocal1 =
            this->tmpBuf1.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmpLocal2 =
            this->tmpBuf2.Get<DTYPE_X>();

        half scalarMinusOne = (half)-1.0;

        // tmpLocal0 = exp(x)
        AscendC::Exp(
            tmpLocal0,
            xLocal,
            this->tileLength);

        // tmpLocal1 = -x
        AscendC::Muls(
            tmpLocal1,
            xLocal,
            scalarMinusOne,
            this->tileLength);

        // tmpLocal1 = exp(-x)
        AscendC::Exp(
            tmpLocal1,
            tmpLocal1,
            this->tileLength);

        // yLocal = exp(x) - exp(-x)
        AscendC::Sub(
            yLocal,
            tmpLocal0,
            tmpLocal1,
            this->tileLength);

        // tmpLocal2 = exp(x) + exp(-x)
        AscendC::Add(
            tmpLocal2,
            tmpLocal0,
            tmpLocal1,
            this->tileLength);

        // y = 分子 / 分母
        AscendC::Div(
            yLocal,
            yLocal,
            tmpLocal2,
            this->tileLength);

        this->outQueueY
            .EnQue<DTYPE_Y>(yLocal);

        this->inQueueX
            .FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            this->outQueueY
                .DeQue<DTYPE_Y>();

        AscendC::DataCopy(
            this->yGm[
                progress *
                this->tileLength],
            yLocal,
            this->tileLength);

        this->outQueueY
            .FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> outQueueY;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC>
        tmpBuf0, tmpBuf1, tmpBuf2;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__
void tanh_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        TanhCustomTilingData);

    GET_TILING_DATA(
        tilingData,
        tiling);

    KernelTanh op;

    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum);

    op.Process();
}