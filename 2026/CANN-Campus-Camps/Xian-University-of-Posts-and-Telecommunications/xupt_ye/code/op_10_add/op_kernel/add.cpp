// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        uint32_t blockNum = AscendC::GetBlockNum();

        this->blockLength = totalLength / blockNum;
        this->tileNum = tileNum;

        uint32_t baseTileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->tileLength = baseTileLength > 0 ? baseTileLength : this->blockLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockIdx * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockIdx * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockIdx * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = this->tileNum * BUFFER_NUM;

        if (loopCount == 0) {
            return;
        }

        //预取第一块
        CopyIn(0);
        //流水线：搬入下一块，计算前一块
        for (uint32_t i = 1; i < loopCount; i++) {
            CopyIn(i);
            Compute(i - 1);
        }
        //处理最后一块
        Compute(loopCount - 1);
        //统一写回所有结果
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t copyLen = ((offset + tileLength) > blockLength) ? (blockLength - offset) : tileLength;

        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm[offset], copyLen);
        AscendC::DataCopy(yLocal, yGm[offset], copyLen);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t calcLen = ((offset + tileLength) > blockLength) ? (blockLength - offset) : tileLength;

        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Add(zLocal, xLocal, yLocal, calcLen);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t writeLen = ((offset + tileLength) > blockLength) ? (blockLength - offset) : tileLength;

        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, writeLen);
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

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);

    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}