#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

// 算子仅支持 Float16 输入输出；若工程编译时未通过编译参数注入 DTYPE_X/DTYPE_Y，则在此兜底定义
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
        // 1. 多核切分：每个核分到等长的数据块 blockLength（与 add_example 的 blockFactor 切分思路一致）
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        // 2. UB 切分：每核数据先按 tileNum 切成超块，超块再按 BUFFER_NUM 双缓冲切成小块
        //    Process() 共循环 tileNum * BUFFER_NUM 次，每次处理 tileLength 个元素
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 3. 绑定 Global Memory 上本核负责的输入/输出区间
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), this->blockLength);

        // 4. 申请队列双缓冲内存（输入队列/输出队列）及计算用的临时 buffer（tmpBuf0~2）
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
        // 从 Global Memory 搬运第 progress 块（tileLength 个元素）到输入队列缓冲区
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> tmp0 = tmpBuf0.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmp1 = tmpBuf1.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmp2 = tmpBuf2.Get<DTYPE_X>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        Exp(tmp0, xLocal, this->tileLength);                            // tmp0 = exp(x)
        Muls(tmp1, xLocal, (DTYPE_X)-1.0f, this->tileLength);           // tmp1 = -x
        Exp(tmp1, tmp1, this->tileLength);                              // tmp1 = exp(-x)
        Add(tmp2, tmp0, tmp1, this->tileLength);                        // tmp2 = exp(x) + exp(-x)
        Sub(tmp0, tmp0, tmp1, this->tileLength);                        // tmp0 = exp(x) - exp(-x)
        Div(yLocal, tmp0, tmp2, this->tileLength);                      // y = tmp0 / tmp2

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取计算结果并搬回 Global Memory
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 读取 host 侧 TilingFunc 下发的切分参数，完成算子初始化和计算
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
