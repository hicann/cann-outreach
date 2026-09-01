#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 1. 合法性校验：blockDim 与 tileNum 不能为 0
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        ASSERT(tileNum != 0 && "tile num can not be zero!");

        // 2. 计算每个核 / 每个 tile 需处理的元素个数
        //    blockLength: 每个核处理的总元素数
        //    tileLength : 每个 tile 处理的元素数（含 double-buffer, 故除以 BUFFER_NUM）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 3. 设置 GM 地址：每个核只处理属于自己的 blockLength 段
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);

        // 4. 初始化队列与临时 Buffer
        //    inQueueX / outQueueY 使用 double-buffer 以掩盖 MTE2/MTE3 与 AICore 计算时延
        //    tmpBuf0/1/2 分别用于存放 exp(x)、exp(-x)、分子 (exp(x) - exp(-x))
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        // double-buffer 下总循环次数 = tileNum * BUFFER_NUM
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
        // 从 GM 搬运一段 tile 数据到 Local（UB）
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 实现 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 复用三个临时 Buffer: expPos=exp(x), expNeg=exp(-x), numerator=分子
        AscendC::LocalTensor<DTYPE_X> expPos   = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNeg   = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> numerator = tmpBuf2.Get<DTYPE_X>();

        const DTYPE_X negOne = -1;
        AscendC::Exp(expPos, xLocal, this->tileLength);                  // expPos = exp(x)
        AscendC::Muls(expNeg, xLocal, negOne, this->tileLength);         // expNeg = -x
        AscendC::Exp(expNeg, expNeg, this->tileLength);                  // expNeg = exp(-x)
        AscendC::Sub(numerator, expPos, expNeg, this->tileLength);       // numerator = exp(x) - exp(-x)
        AscendC::Add(expPos, expPos, expNeg, this->tileLength);          // expPos = exp(x) + exp(-x)  (分母, 复用)
        AscendC::Div(yLocal, numerator, expPos, this->tileLength);       // y = numerator / 分母

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从 Local 回写到 GM 对应位置
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 实例化算子并执行
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
