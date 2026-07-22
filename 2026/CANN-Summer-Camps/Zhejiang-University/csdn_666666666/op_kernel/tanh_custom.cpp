#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BLOCK_DIM = 8;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        this->tileNum = tileNum;
        this->blockLength = (totalLength + BLOCK_DIM - 1) / BLOCK_DIM;
        this->tileLength = (this->blockLength + tileNum - 1) / tileNum;
        int64_t offset = this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x) + offset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(y) + offset, this->blockLength);

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
        // TODO: 考生自行补齐
        int32_t tileIdx = progress / BUFFER_NUM;
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = this->tileLength * sizeof(DTYPE_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm[tileIdx * this->tileLength], copyParams, {false, 0, 0, 0});
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // tmpBuf0 = exp(x)
        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();
        AscendC::Exp(expX, xLocal, this->tileLength);

        // tmpBuf1 = -x
        AscendC::LocalTensor<DTYPE_X> negX = tmpBuf1.Get<DTYPE_X>();
        AscendC::Muls(negX, xLocal, (half)(-1.0), this->tileLength);

        // tmpBuf1 = exp(-x)
        AscendC::Exp(negX, negX, this->tileLength);

        // tmpBuf2 = exp(x) - exp(-x)
        AscendC::LocalTensor<DTYPE_X> numerator = tmpBuf2.Get<DTYPE_X>();
        AscendC::Sub(numerator, expX, negX, this->tileLength);

        // negX = exp(x) + exp(-x)  复用 tmpBuf1
        AscendC::Add(negX, expX, negX, this->tileLength);

        // yLocal = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, numerator, negX, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        int32_t tileIdx = progress / BUFFER_NUM;
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = this->tileLength * sizeof(DTYPE_Y);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[tileIdx * this->tileLength], yLocal, copyParams);
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
    op.Init(x, y, static_cast<uint32_t>(tilingData.totalNum), TILE_NUM);
    op.Process();
}