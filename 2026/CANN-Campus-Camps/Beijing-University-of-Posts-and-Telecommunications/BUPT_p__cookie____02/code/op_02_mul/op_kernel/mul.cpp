// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tilingData) {
        this->blockIndex = GetBlockIdx();
        this->blockLength = tilingData.tileLength;
        if (this->blockIndex == tilingData.tileNum - 1) {
            this->blockLength = tilingData.lastTileLength;
        }
        uint32_t offset = this->blockIndex * tilingData.tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        CopyIn(0);
        Compute(0);
        CopyOut(0);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm, this->blockLength);
        DataCopy(yLocal, yGm, this->blockLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        Mul(zLocal, xLocal, yLocal, this->blockLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm, zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t blockIndex;
    uint32_t blockLength;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
