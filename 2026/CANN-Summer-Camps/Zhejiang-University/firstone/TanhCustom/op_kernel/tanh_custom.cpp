#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
       // 1. 设置切分参数[cite: 6]
        this->tileNum = tileNum;
        
        // 计算当前物理核 (Block) 需要处理的总数据量
        // GetBlockNum() 获取总核数（如 8 核），GetBlockIdx() 获取当前是第几个核
        this->blockLength = totalLength / AscendC::GetBlockNum();
        
        // 计算流水线中每次处理的切片大小 (Tile Length)
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 2. 初始化全局内存张量 (GlobalTensor)，注意偏移量计算[cite: 6]
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 3. 初始化流水线通信队列 (TPipe)[cite: 6]
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));

        // 4. 初始化用于复杂数学公式计算的临时 Buffer[cite: 6]
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 1. 取出输入数据并分配输出内存[cite: 6]
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        // 2. 申请临时 Tensor 用于拆解 Tanh 公式[cite: 6]
        AscendC::LocalTensor<DTYPE_X> tmp0 = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp1 = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp2 = tmpBuf2.Get<DTYPE_X>();

        // 3. 严格遵循 API 约束，无基础运算符、强制转换 half、绝对没有变量 z
        // 步骤 a: tmp2 = -x  (乘法缩放，使用 half 强转标量)
        AscendC::Muls(tmp2, xLocal, (half)-1.0, this->tileLength);
        
        // 步骤 b: tmp0 = exp(x)
        AscendC::Exp(tmp0, xLocal, this->tileLength);
        
        // 步骤 c: tmp1 = exp(-x)
        AscendC::Exp(tmp1, tmp2, this->tileLength);
        
        // 步骤 d: tmp2 = exp(x) - exp(-x)  (分子)
        AscendC::Sub(tmp2, tmp0, tmp1, this->tileLength);
        
        // 步骤 e: yLocal = exp(x) + exp(-x) (分母，直接写入 yLocal 省内存)
        AscendC::Add(yLocal, tmp0, tmp1, this->tileLength);
        
        // 步骤 f: yLocal = tmp2 / yLocal  (分子除以分母)
        AscendC::Div(yLocal, tmp2, yLocal, this->tileLength);

        // 4. 将计算结果推入输出队列，并释放输入 Tensor[cite: 6]
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
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
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();

}