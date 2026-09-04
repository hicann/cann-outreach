// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t QUEUE_DEPTH = 2; // 深度 2：搬入/计算/搬出三段流水重叠，隐藏计算与搬出延迟

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        // 每个核处理的总数据长度（多核均分）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        // 单核内部切 tileNum 块，每块长度 tileLength（单块方案 tileNum=1）
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;
        // 设置每个核的 Global Memory 起始地址（多核处理的数据段互不重叠）
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 初始化队列缓冲
        pipe.InitBuffer(inQueueX, QUEUE_DEPTH, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, QUEUE_DEPTH, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, QUEUE_DEPTH, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        // 循环次数 = tileNum（配合深度 2 的队列形成双缓冲流水）
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 为输入 LocalTensor 分配内存
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        // Global Memory -> Local Memory
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        // 数据入队，交给 Compute 处理
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从队首取出输入数据
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        // 为输出 LocalTensor 分配内存
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法指令：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        // 结果入队，释放输入
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从队首取出输出结果
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        // Local Memory -> Global Memory
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        // 释放输出内存
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength; // 每核计算数据长度
    uint32_t tileNum;     // 每核内部分块数
    uint32_t tileLength;  // 每块长度
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
