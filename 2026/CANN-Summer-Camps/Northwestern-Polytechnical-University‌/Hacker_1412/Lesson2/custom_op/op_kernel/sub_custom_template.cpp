#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();

        const uint32_t baseLength = totalLength / blockNum;
        const uint32_t remainder = totalLength % blockNum;
        this->blockLength = baseLength + (blockIdx < remainder ? 1U : 0U);
        this->blockOffset = blockIdx * baseLength + (blockIdx < remainder ? blockIdx : remainder);

        if (this->blockLength == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        DataCopy(xLocal, xGm[0], this->blockLength);
        DataCopy(yLocal, yGm[0], this->blockLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        Sub(zLocal, xLocal, yLocal, this->blockLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        DataCopy(zGm[0], zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength = 0;
    uint32_t blockOffset = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    KernelSubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
