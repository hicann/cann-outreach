#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength)
    {
        blockLength = totalLength / AscendC::GetBlockNum();

        uint32_t blockOffset =
            blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X *)x + blockOffset,
            blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y *)y + blockOffset,
            blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DTYPE_Z *)z + blockOffset,
            blockLength);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            blockLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            blockLength * sizeof(DTYPE_Y));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            blockLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueueX.AllocTensor<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_Y> yLocal =
            inQueueY.AllocTensor<DTYPE_Y>();

        AscendC::DataCopy(
            xLocal,
            xGm,
            blockLength);

        AscendC::DataCopy(
            yLocal,
            yGm,
            blockLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueueX.DeQue<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_Y> yLocal =
            inQueueY.DeQue<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_Z> zLocal =
            outQueueZ.AllocTensor<DTYPE_Z>();

        AscendC::Sub(
            zLocal,
            xLocal,
            yLocal,
            blockLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal =
            outQueueZ.DeQue<DTYPE_Z>();

        AscendC::DataCopy(
            zGm,
            zLocal,
            blockLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;

    uint32_t blockLength = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}