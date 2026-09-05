// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_LENGTH = 2048;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        uint32_t blockOffset = BLOCK_LENGTH * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, BLOCK_LENGTH);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, BLOCK_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, BLOCK_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, BLOCK_LENGTH * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm, BLOCK_LENGTH);
        AscendC::DataCopy(yLocal, yGm, BLOCK_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, BLOCK_LENGTH);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm, zLocal, BLOCK_LENGTH);
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
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    KernelMul<DT_X> op;
    op.Init(x, y, z);
    op.Process();
}
