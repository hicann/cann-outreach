#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = GetBlockNum();
        ASSERT(blockNum != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / blockNum;
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        uint32_t blockIdx = GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ half*)x + blockIdx * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + blockIdx * this->blockLength, this->blockLength);

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
        uint32_t tileIdx = progress % this->tileNum;
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[tileIdx * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> expPos = tmpBuf0.Get<half>();
        LocalTensor<half> expNeg = tmpBuf1.Get<half>();
        LocalTensor<half> tmp = tmpBuf2.Get<half>();

        // exp(x)
        Exp(expPos, xLocal, this->tileLength);
        // exp(-x): 先取负再 Exp
        half negOne = -1.0f;
        Muls(expNeg, xLocal, negOne, this->tileLength);
        Exp(expNeg, expNeg, this->tileLength);

        // 分子 = exp(x) - exp(-x)
        Sub(tmp, expPos, expNeg, this->tileLength);
        // 分母 = exp(x) + exp(-x)，复用 expPos 存储分母
        Add(expPos, expPos, expNeg, this->tileLength);
        // 结果 = 分子 / 分母
        Div(tmp, tmp, expPos, this->tileLength);

        // 将结果放入输出队列
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        DataCopy(yLocal, tmp, this->tileLength);
        outQueueY.EnQue(yLocal);

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t tileIdx = progress % this->tileNum;
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[tileIdx * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalElements, tilingData.tileNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
