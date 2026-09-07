#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t blockLength, AscendC::TPipe *pipe)
    {
        this->blockLength = blockLength;
        const uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, blockLength);

        pipe->InitBuffer(inQueueX, 1, blockLength * sizeof(DT_X));
        pipe->InitBuffer(inQueueY, 1, blockLength * sizeof(DT_X));
        pipe->InitBuffer(outQueueZ, 1, blockLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm, blockLength);
        AscendC::DataCopy(yLocal, yGm, blockLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, blockLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(zGm, zLocal, blockLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    AscendC::TPipe pipe;
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.blockLength, &pipe);
    op.Process();
}