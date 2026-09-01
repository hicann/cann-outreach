#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using DTYPE_X = half;
using DTYPE_Y = half;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->tileNum = tileNum;
        this->blockLength = totalLength;
        this->tileLength = totalLength / tileNum;

        // 根据core ID计算偏移，每个core处理不同的数据段
        uint32_t blockOffset = AscendC::GetBlockIdx() * blockLength * BUFFER_NUM;

        this->xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + blockOffset, blockLength * BUFFER_NUM);
        this->yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + blockOffset, blockLength * BUFFER_NUM);

        // 为输入队列分配LocalTensor
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        // 为输出队列分配LocalTensor
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        // 为中间计算分配临时Buffer
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(half));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy<DTYPE_X>(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 取出输入tensor
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 获取临时buffer用于中间计算
        AscendC::LocalTensor<half> expXBuf = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> expNegXBuf = tmpBuf1.Get<half>();
        AscendC::LocalTensor<half> tmpBuf = tmpBuf2.Get<half>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // 1. 计算 -x
        AscendC::Muls(tmpBuf, xLocal, (half)(-1.0f), tileLength);
        // 2. 计算 exp(x)
        AscendC::Exp(expXBuf, xLocal, tileLength);
        // 3. 计算 exp(-x)
        AscendC::Exp(expNegXBuf, tmpBuf, tileLength);
        // 4. 计算 exp(x) + exp(-x) -> 存入tmpBuf
        AscendC::Add(tmpBuf, expXBuf, expNegXBuf, tileLength);
        // 5. 计算 exp(x) - exp(-x) -> 存入expXBuf
        AscendC::Sub(expXBuf, expXBuf, expNegXBuf, tileLength);
        // 6. 计算 (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, expXBuf, tmpBuf, tileLength);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy<DTYPE_Y>(yGm[progress * tileLength], yLocal, tileLength);
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