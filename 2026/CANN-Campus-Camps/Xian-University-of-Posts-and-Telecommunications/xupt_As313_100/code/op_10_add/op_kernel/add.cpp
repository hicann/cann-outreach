// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t formerNum,
                                uint32_t formerLength, uint32_t tailLength, uint32_t tileLength) {
        // 前formerNum个核处理大块, 其余核处理小块, 各核负载尽量均衡
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < formerNum) {
            this->coreLength = formerLength;
            this->coreOffset = blockIdx * formerLength;
        } else {
            this->coreLength = tailLength;
            this->coreOffset = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
        }
        this->tileLength = tileLength;
        this->loopCount = 0;
        this->lastTileLength = tileLength;
        if (this->coreLength == 0) {
            return;
        }
        // 循环参数在Init里一次算好, 避免Process内做除法
        this->loopCount = (this->coreLength + this->tileLength - 1) / this->tileLength;
        this->lastTileLength = this->coreLength - (this->loopCount - 1) * this->tileLength;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->coreOffset, this->coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + this->coreOffset, this->coreLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            // 尾块可能不满一个tile, 其余都是整块
            this->curLength = (i == this->loopCount - 1) ? this->lastTileLength : this->tileLength;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        // x、y两条搬运紧邻发射, 中间无依赖, 在MTE2流水线上排队
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->curLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Add(zLocal, xLocal, yLocal, this->curLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->curLength);
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
    uint32_t coreOffset;      // 本核数据在GM上的起始偏移
    uint32_t coreLength;      // 本核处理的元素个数
    uint32_t tileLength;      // 核内单次处理的元素个数
    uint32_t curLength;       // 当前块的实际长度
    uint32_t lastTileLength;  // 尾块长度
    int32_t loopCount;        // 核内循环次数
};

template <typename DT_X>
 __global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.formerNum, tiling_data.formerLength,
            tiling_data.tailLength, tiling_data.tileLength);
    op.Process();
}
