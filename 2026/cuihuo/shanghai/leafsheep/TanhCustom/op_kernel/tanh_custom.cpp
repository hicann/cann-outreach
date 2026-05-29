#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
    // 1. 计算当前 Block 负责的数据范围
    this->blockLength = totalLength / AscendC::GetBlockNum();
    uint32_t offset   = AscendC::GetBlockIdx() * this->blockLength;

    // 2. 绑定全局内存地址
    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + offset, this->blockLength);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, this->blockLength);

    // 3. 保存 Tile 参数
    this->tileNum    = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

    // 4. 分配片上内存
    pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    pipe.InitBuffer(tmpBuf0,   this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(tmpBuf1,   this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(tmpBuf2,   this->tileLength * sizeof(DTYPE_X)); 
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
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        // 取出输入，申请输出
    AscendC::LocalTensor<DTYPE_X> xLocal  = inQueueX.DeQue<DTYPE_X>();
    AscendC::LocalTensor<DTYPE_Y> yLocal  = outQueueY.AllocTensor<DTYPE_Y>();

    // 取出中间缓冲区
    AscendC::LocalTensor<DTYPE_X> expPos  = tmpBuf0.Get<DTYPE_X>();  // 存 exp(x)
    AscendC::LocalTensor<DTYPE_X> expNeg  = tmpBuf1.Get<DTYPE_X>();  // 存 exp(-x)
    AscendC::LocalTensor<DTYPE_X> tmp     = tmpBuf2.Get<DTYPE_X>();  // 存分子

    // 计算 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // Step 1: expPos = exp(x)
    AscendC::Exp(expPos, xLocal, this->tileLength);

    // Step 2: expNeg = -x
    AscendC::Muls(expNeg, xLocal, (half)-1.0, this->tileLength);

    // Step 3: expNeg = exp(-x)
    AscendC::Exp(expNeg, expNeg, this->tileLength);

    // Step 4: tmp = exp(x) - exp(-x)  （分子）
    AscendC::Sub(tmp, expPos, expNeg, this->tileLength);

    // Step 5: yLocal = exp(x) + exp(-x)  （分母，借用 yLocal 暂存）
    AscendC::Add(yLocal, expPos, expNeg, this->tileLength);

    // Step 6: yLocal = 分子 / 分母 = tanh(x)
    AscendC::Div(yLocal, tmp, yLocal, this->tileLength);

    // 释放输入，输出入队
    inQueueX.FreeTensor(xLocal);
    outQueueY.EnQue(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
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
    // TODO: 考生自行补齐
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
