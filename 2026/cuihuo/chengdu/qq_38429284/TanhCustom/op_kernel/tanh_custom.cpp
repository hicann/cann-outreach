#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr int32_t BLOCK_DIM = 8;  // 核数，与 host 侧 BLOCK_DIM 保持一致

// 若框架未注入数据类型宏，则默认 float16
#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        ASSERT(GetBlockIdx() < BLOCK_DIM);
        this->blockLength = totalLength / BLOCK_DIM;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / (this->tileNum * BUFFER_NUM);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * GetBlockIdx(), this->blockLength);

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
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();      // exp(x)
        LocalTensor<DTYPE_X> expNegX = tmpBuf1.Get<DTYPE_X>();   // exp(-x)
        LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();       // 临时空间

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        Exp(expX, xLocal, this->tileLength);                              // expX = exp(x)
        Muls(tmp, xLocal, (DTYPE_X)(-1.0), this->tileLength);              // tmp = -x
        Exp(expNegX, tmp, this->tileLength);                               // expNegX = exp(-x)
        Sub(yLocal, expX, expNegX, this->tileLength);                      // yLocal = 分子
        Add(tmp, expX, expNegX, this->tileLength);                         // tmp = 分母
        Div(yLocal, yLocal, tmp, this->tileLength);                        // y = 分子 / 分母

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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
    (void)workspace;
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: 考生自行补齐
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}