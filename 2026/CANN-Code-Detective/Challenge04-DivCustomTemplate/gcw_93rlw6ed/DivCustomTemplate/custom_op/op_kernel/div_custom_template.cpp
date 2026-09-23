#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // double buffering

class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        uint32_t totalSize = tiling_data.size;
        uint32_t blockDim = GetBlockNum();
        uint32_t coreIdx = GetBlockIdx();

        // 每个核处理的数据量，最后一个核处理余数
        tileLength = totalSize / blockDim;
        tileOffset = coreIdx * tileLength;
        if (coreIdx == blockDim - 1) {
            tileLength += totalSize % blockDim;
        }

        xGlobal.SetGlobalBuffer((__gm__ half *)x + tileOffset, tileLength);
        yGlobal.SetGlobalBuffer((__gm__ half *)y + tileOffset, tileLength);
        zGlobal.SetGlobalBuffer((__gm__ half *)z + tileOffset, tileLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
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
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGlobal, tileLength);
        DataCopy(yLocal, yGlobal, tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Div(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGlobal, zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGlobal, yGlobal, zGlobal;
    uint32_t tileLength;
    uint32_t tileOffset;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDiv op;
    op.Init(x, y, z, tiling);
    op.Process();
}
