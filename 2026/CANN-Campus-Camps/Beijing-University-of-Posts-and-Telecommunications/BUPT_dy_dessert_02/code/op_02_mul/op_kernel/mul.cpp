// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,uint32_t length,uint32_t tileNum) {
        this->blockLength = length / AscendC::GetBlockNum();
        // 单核内分块数，类似几个 batch
        this->tileNum = tileNum;
        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存（单核内分块处理的基础）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for(int32_t i=0;i<loopCount;i++){
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
    
private:
    AscendC::TPipe pipe;  // TPipe 内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  // 输入数据队列管理对象，VECIN 位置
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;  // 输出数据队列管理对象，VECOUT 位置
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm; // 管理输入输出 Global Memory 地址的对象，其中 xGm、yGm 为输入，zGm 为输出

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数，类似 batch 数量
    uint32_t tileLength;  // 单核内每块元素数，类似 batch size
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length,tiling_data.tileNum);
    op.Process();
}
