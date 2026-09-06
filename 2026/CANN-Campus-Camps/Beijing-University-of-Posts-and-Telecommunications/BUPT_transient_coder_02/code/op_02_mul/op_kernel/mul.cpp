// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tilingData) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < tilingData.formerNum) {
            this->curLength = tilingData.formerLength;
            this->curOffset = blockIdx * tilingData.formerLength;
        } else {
            this->curLength = tilingData.tailLength;
            this->curOffset = tilingData.formerNum * tilingData.formerLength +
                              (blockIdx - tilingData.formerNum) * tilingData.tailLength;
        }

        if (this->curLength == 0) {
            return;
        }

        this->alignNum = tilingData.alignNum;
        if (this->alignNum == 0) {
            this->alignNum = 32 / sizeof(DT_X);
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->curOffset, this->curLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->curOffset, this->curLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + this->curOffset, this->curLength);

        this->tileLength = tilingData.tileLength;
        if (this->tileLength > this->curLength || this->tileLength == 0) {
            this->tileLength = this->curLength;
        }

        this->loopCount = (this->curLength + this->tileLength - 1) / this->tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->curLength == 0) {
            return;
        }

        if (this->loopCount == 1) {
            CopyIn(0, this->curLength);
            Compute(this->curLength);
            CopyOut(0, this->curLength);
            return;
        }

        // 双缓冲流水线 (Double Buffering Pipeline)
        // 1. Prologue: 异步预取第 0 块到 Buffer 0 (MTE2)
        CopyIn(0, this->tileLength);

        // 2. Main Loop: 搬入 Tile i+1 (MTE2)、计算 Tile i (Vector)、写回 Tile i-1 (MTE3) 全流水重叠
        for (uint32_t i = 0; i < this->loopCount - 1; i++) {
            uint32_t nextOffset = (i + 1) * this->tileLength;
            uint32_t nextLen = (i + 1 == this->loopCount - 1) ?
                               (this->curLength - nextOffset) : this->tileLength;

            // 异步发起下一块搬入 (MTE2)
            CopyIn(nextOffset, nextLen);

            // 执行当前块计算 (Vector)
            Compute(this->tileLength);

            // 异步写回当前块结果 (MTE3)
            CopyOut(i * this->tileLength, this->tileLength);
        }

        // 3. Epilogue: 计算并写回最后一块
        uint32_t lastOffset = (this->loopCount - 1) * this->tileLength;
        uint32_t lastLen = this->curLength - lastOffset;
        Compute(lastLen);
        CopyOut(lastOffset, lastLen);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t actualLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        uint32_t copyLength = (actualLength + this->alignNum - 1) & ~(this->alignNum - 1);
        AscendC::DataCopy(xLocal, xGm[offset], copyLength);
        AscendC::DataCopy(yLocal, yGm[offset], copyLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t actualLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        uint32_t calcLength = (actualLength + this->alignNum - 1) & ~(this->alignNum - 1);

        AscendC::Mul(zLocal, xLocal, yLocal, calcLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t actualLength)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        uint32_t copyLength = (actualLength + this->alignNum - 1) & ~(this->alignNum - 1);
        AscendC::DataCopy(zGm[offset], zLocal, copyLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t curLength = 0;
    uint32_t curOffset = 0;
    uint32_t tileLength = 0;
    uint32_t loopCount = 0;
    uint32_t alignNum = 8;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
