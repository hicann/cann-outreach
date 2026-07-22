#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0);
        ASSERT(tileNum != 0);

        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        const uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(x) + blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(y) + blockOffset,
            this->blockLength);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(half));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(half));
        
        pipe.InitBuffer(
            tmpBuf0,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            tmpBuf1,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            tmpBuf2,
            this->tileLength * sizeof(DTYPE_X));
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
        AscendC::LocalTensor<DTYPE_X> xLocal =
        inQueueX.AllocTensor<DTYPE_X>();

        const uint32_t offset = progress * tileLength;

        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            tileLength);

        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
        inQueueX.DeQue<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_Y> yLocal =
        outQueueY.AllocTensor<DTYPE_Y>();

    AscendC::LocalTensor<DTYPE_X> tmp0 =
        tmpBuf0.Get<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_X> tmp1 =
        tmpBuf1.Get<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_X> tmp2 =
        tmpBuf2.Get<DTYPE_X>();

    AscendC::Muls(
        tmp0,
        xLocal,
        static_cast<DTYPE_X>(-1.0f),
        tileLength);

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Exp(
        tmp1,
        xLocal,
        tileLength);

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Exp(
        tmp2,
        tmp0,
        tileLength);

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Sub(
        tmp0,
        tmp1,
        tmp2,
        tileLength);

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Add(
        tmp1,
        tmp1,
        tmp2,
        tileLength);

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div(
        yLocal,
        tmp0,
        tmp1,
        tileLength);

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        const uint32_t offset = static_cast<uint32_t>(progress) * this->tileLength;

        AscendC::DataCopy(
            yGm[offset],
            yLocal,
            tileLength);

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
    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum);

    op.Process();

}