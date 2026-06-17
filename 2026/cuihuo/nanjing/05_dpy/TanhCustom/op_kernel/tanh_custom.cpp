#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        // 1. 计算切分参数：获取每个 Block 需要处理的数据量，以及每个 Tile 的数据量
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 框架代码中 loopCount = this->tileNum * BUFFER_NUM
        // 说明单核数据被切分为 tileNum * BUFFER_NUM 份
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 2. 初始化 Global Tensor 的地址偏移
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 3. 为队列和临时缓存分配内存空间
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
        AscendC::LocalTensor<DTYPE_X> inLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(inLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(inLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        AscendC::LocalTensor<DTYPE_X> inLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        // 获取三个临时 Buffer
        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNegX = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> numerator = tmpBuf2.Get<DTYPE_X>();

        // 1. 计算 exp(x)
        AscendC::Exp(expX, inLocal, this->tileLength);

        // 2. 计算 exp(-x) -> 先乘 -1.0，再求 Exp
        AscendC::Muls(expNegX, inLocal, (DTYPE_X)(-1.0), this->tileLength);
        AscendC::Exp(expNegX, expNegX, this->tileLength);

        // 3. 计算分子: exp(x) - exp(-x)
        AscendC::Sub(numerator, expX, expNegX, this->tileLength);

        // 4. 计算分母: exp(x) + exp(-x)，直接复用 outLocal 暂存以节省内存
        AscendC::Add(outLocal, expX, expNegX, this->tileLength);

        // 5. 计算最终结果 y = 分子 / 分母
        AscendC::Div(outLocal, numerator, outLocal, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(outLocal);
        inQueueX.FreeTensor(inLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueY.FreeTensor(outLocal);
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
    // 初始化类并调用计算流程
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}