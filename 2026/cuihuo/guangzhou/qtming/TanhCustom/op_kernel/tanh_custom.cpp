#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 每个核处理的总数据长度（host侧已按核数切分好）
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        // 每个tile处理的数据长度：双缓冲，每个队列分配BUFFER_NUM个buffer
        this->tileLength = totalLength / (tileNum * BUFFER_NUM);

        // 每个核负责处理 [GetBlockIdx() * blockLength, (GetBlockIdx() + 1) * blockLength) 的数据段
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + AscendC::GetBlockIdx() * this->blockLength,
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + AscendC::GetBlockIdx() * this->blockLength,
                            this->blockLength);

        // 初始化输入/输出队列buffer（双缓冲），以及3个中间计算buffer
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
        // 从GlobalMemory搬入一个tile的输入数据到LocalMemory
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();   // exp(x)
        AscendC::LocalTensor<DTYPE_X> expNX = tmpBuf1.Get<DTYPE_X>();  // exp(-x)
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();    // 中间结果

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // tmp = -x
        AscendC::Muls(tmp, xLocal, (DTYPE_X)(-1.0f), this->tileLength);
        // expX = exp(x), expNX = exp(-x)
        AscendC::Exp(expX, xLocal, this->tileLength);
        AscendC::Exp(expNX, tmp, this->tileLength);
        // tmp = exp(x) - exp(-x)
        AscendC::Sub(tmp, expX, expNX, this->tileLength);
        // expX = exp(x) + exp(-x)
        AscendC::Add(expX, expX, expNX, this->tileLength);
        // y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, tmp, expX, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 将计算结果从LocalMemory搬出到GlobalMemory
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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
    // 实例化算子类，Init完成tiling参数解析与buffer初始化，Process完成搬入/计算/搬出流水
    KernelTanh op;
    op.Init(x, y, tilingData.blockLength, tilingData.tileNum);
    op.Process();
}