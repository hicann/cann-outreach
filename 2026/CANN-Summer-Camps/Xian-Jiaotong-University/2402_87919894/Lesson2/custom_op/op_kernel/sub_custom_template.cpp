#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength)
    {
        blockLength = totalLength / GetBlockNum();

        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X *)x + blockLength * GetBlockIdx(),
            blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y *)y + blockLength * GetBlockIdx(),
            blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DTYPE_Z *)z + blockLength * GetBlockIdx(),
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
        LocalTensor<DTYPE_X> xLocal =
            inQueueX.AllocTensor<DTYPE_X>();

        LocalTensor<DTYPE_Y> yLocal =
            inQueueY.AllocTensor<DTYPE_Y>();

        DataCopy(xLocal, xGm, blockLength);
        DataCopy(yLocal, yGm, blockLength);

        inQueueX.EnQue<DTYPE_X>(xLocal);
        inQueueY.EnQue<DTYPE_Y>(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DTYPE_X> xLocal =
            inQueueX.DeQue<DTYPE_X>();

        LocalTensor<DTYPE_Y> yLocal =
            inQueueY.DeQue<DTYPE_Y>();

        LocalTensor<DTYPE_Z> zLocal =
            outQueueZ.AllocTensor<DTYPE_Z>();

        Sub(zLocal, xLocal, yLocal, blockLength);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<DTYPE_Z> zLocal =
            outQueueZ.DeQue<DTYPE_Z>();

        DataCopy(zGm, zLocal, blockLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;

    uint32_t blockLength;
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
