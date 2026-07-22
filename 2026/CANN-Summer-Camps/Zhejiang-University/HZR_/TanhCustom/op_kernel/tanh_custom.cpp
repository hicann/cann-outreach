#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        this->tileNum = tileNum;

        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        uint32_t blockOffset =
            this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_X*>(x) + blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_Y*>(y) + blockOffset,
            this->blockLength);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_Y));

pipe.InitBuffer(
    tmpBuf0,
    this->tileLength * sizeof(float));

pipe.InitBuffer(
    tmpBuf1,
    this->tileLength * sizeof(float));

pipe.InitBuffer(
    tmpBuf2,
    this->tileLength * sizeof(float));
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

        uint32_t offset = progress * this->tileLength;

        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            this->tileLength);

        inQueueX.EnQue(xLocal);
    }

__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<DTYPE_X> xLocal =
        inQueueX.DeQue<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_Y> yLocal =
        outQueueY.AllocTensor<DTYPE_Y>();

    // 三个 FP32 中间缓冲区
    AscendC::LocalTensor<float> tmp0 =
        tmpBuf0.Get<float>();

    AscendC::LocalTensor<float> tmp1 =
        tmpBuf1.Get<float>();

    AscendC::LocalTensor<float> tmp2 =
        tmpBuf2.Get<float>();

    // FP16 输入转换为 FP32：tmp0 = x
    AscendC::Cast(
        tmp0,
        xLocal,
        AscendC::RoundMode::CAST_NONE,
        this->tileLength);

    // tmp1 = -x
    AscendC::Muls(
        tmp1,
        tmp0,
        -1.0f,
        this->tileLength);

    // tmp0 = exp(x)
    AscendC::Exp(
        tmp0,
        tmp0,
        this->tileLength);

    // tmp1 = exp(-x)
    AscendC::Exp(
        tmp1,
        tmp1,
        this->tileLength);

    // tmp2 = exp(x) + exp(-x)
    AscendC::Add(
        tmp2,
        tmp0,
        tmp1,
        this->tileLength);

    // tmp0 = exp(x) - exp(-x)
    AscendC::Sub(
        tmp0,
        tmp0,
        tmp1,
        this->tileLength);

    // tmp1 = tanh(x)
    AscendC::Div(
        tmp1,
        tmp0,
        tmp2,
        this->tileLength);

    // FP32 结果转换回 FP16
    AscendC::Cast(
        yLocal,
        tmp1,
        AscendC::RoundMode::CAST_NONE,
        this->tileLength);

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}   

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueueY.DeQue<DTYPE_Y>();

        uint32_t offset = progress * this->tileLength;

        AscendC::DataCopy(
            yGm[offset],
            yLocal,
            this->tileLength);

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
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();

}