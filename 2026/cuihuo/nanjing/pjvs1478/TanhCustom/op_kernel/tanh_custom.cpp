#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->tileNum = tileNum;
        this->blockLength = totalLength;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 获取当前核的block index，计算该核对应的全局内存偏移
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 初始化双缓冲队列：VECIN用于搬入，VECOUT用于搬出
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        // 从队列中分配LocalTensor用于接收数据
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        // 从GlobalMemory搬入数据到LocalTensor
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        // 将数据入队供Compute使用
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列出队
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        // 从输出队列分配LocalTensor
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 获取临时缓冲区，用于中间计算
        AscendC::LocalTensor<DTYPE_X> tmpExpX      = tmpBuf0.Get<DTYPE_X>();  // exp(x)
        AscendC::LocalTensor<DTYPE_X> tmpExpNegX   = tmpBuf1.Get<DTYPE_X>();  // exp(-x)
        AscendC::LocalTensor<DTYPE_X> tmpNumerator = tmpBuf2.Get<DTYPE_X>();  // exp(x) - exp(-x)

        // Step1: tmpExpX = exp(x)
        AscendC::Exp(tmpExpX, xLocal, this->tileLength);

        // Step2: tmpExpNegX = exp(-x)，先计算 -x，再exp
        // 使用 Muls 乘以 -1 实现取反
        AscendC::Muls(tmpExpNegX, xLocal, (DTYPE_X)(-1), this->tileLength);
        AscendC::Exp(tmpExpNegX, tmpExpNegX, this->tileLength);

        // Step3: tmpNumerator = exp(x) - exp(-x)
        AscendC::Sub(tmpNumerator, tmpExpX, tmpExpNegX, this->tileLength);

        // Step4: tmpExpX = exp(x) + exp(-x) （复用tmpExpX缓冲区）
        AscendC::Add(tmpExpX, tmpExpX, tmpExpNegX, this->tileLength);

        // Step5: yLocal = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Div(yLocal, tmpNumerator, tmpExpX, this->tileLength);

        // 结果入队供CopyOut搬出
        outQueueY.EnQue(yLocal);
        // 释放输入tensor
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列出队
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        // 将计算结果从LocalTensor搬出到GlobalMemory
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // 释放输出tensor
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

    // 创建算子实例，传入tiling参数进行初始化并执行
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
