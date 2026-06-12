#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;  // 添加命名空间简化代码

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 修复1: 使用 AscendC::GetBlockIdx() 或直接使用 GetBlockIdx() (因为已有 using namespace)
        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf0,  this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1,  this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf2,  this->tileLength * sizeof(half));
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
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        // 获取输入输出 tensor
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        // 修复2: 正确使用 Duplicate - 第二个参数应该是标量值，不是 tensor
        // 计算 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        // exp(x) -> yLocal (临时存储 exp(x))
        Exp(yLocal, xLocal, this->tileLength);
        
        // 计算 exp(-x)
        // 先取负: tmpBuf0 = -x
        half negOne(-1.0f);
        half zero(0.0f);
        LocalTensor<half> tmpNeg = tmpBuf0.Get<half>();
        // 使用 Muls 实现乘以 -1
        Muls(tmpNeg, xLocal, negOne, this->tileLength);
        
        // 计算 exp(-x) -> tmpBuf1
        LocalTensor<half> tmpExpNeg = tmpBuf1.Get<half>();
        Exp(tmpExpNeg, tmpNeg, this->tileLength);
        
        // 现在:
        // yLocal = exp(x)
        // tmpExpNeg = exp(-x)
        
        // 计算 exp(x) - exp(-x) -> tmpBuf2
        LocalTensor<half> tmpDiff = tmpBuf2.Get<half>();
        Sub(tmpDiff, yLocal, tmpExpNeg, this->tileLength);
        
        // 计算 exp(x) + exp(-x) -> yLocal (复用)
        Add(yLocal, yLocal, tmpExpNeg, this->tileLength);
        
        // 计算最终结果: (exp(x)-exp(-x)) / (exp(x)+exp(-x)) -> tmpDiff
        Div(tmpDiff, tmpDiff, yLocal, this->tileLength);
        
        // 将结果复制到输出队列
        DataCopy(yLocal, tmpDiff, this->tileLength);
        
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    GlobalTensor<half> xGm;  // 修复: 使用具体类型而不是 DTYPE_X
    GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    KernelTanh op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
