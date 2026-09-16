// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

using namespace AscendC;

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength) {
        (void)tileLength;

        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        constexpr uint32_t ALIGN_UNIT = 32;
        uint32_t totalBlocks = totalLength / ALIGN_UNIT;
        uint32_t blocksPerCore = totalBlocks / blockNum;
        uint32_t remainderBlocks = totalBlocks % blockNum;
        uint32_t thisBlocks = (blockId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        uint32_t thisLength = thisBlocks * ALIGN_UNIT;
        uint32_t offset = (blockId < remainderBlocks)
                              ? (blockId * (blocksPerCore + 1) * ALIGN_UNIT)
                              : (remainderBlocks * (blocksPerCore + 1) + (blockId - remainderBlocks) * blocksPerCore) * ALIGN_UNIT;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, thisLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, thisLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, thisLength);

        thisLength_ = thisLength;

        uint32_t bufBytes = thisLength * sizeof(DT_X);
        pipe.InitBuffer(inQueueX, 1, bufBytes);
        pipe.InitBuffer(inQueueY, 1, bufBytes);
        pipe.InitBuffer(outQueueZ, 1, bufBytes);
    }

    __aicore__ inline void Process() {
        CopyIn(thisLength_);
        Compute(thisLength_);
        CopyOut(thisLength_);
    }

private:
    __aicore__ inline void CopyIn(uint32_t len) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm, len);
        DataCopy(yLocal, yGm, len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t len) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm, zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;
    uint32_t thisLength_;
};

template <typename DT_X>
__global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}