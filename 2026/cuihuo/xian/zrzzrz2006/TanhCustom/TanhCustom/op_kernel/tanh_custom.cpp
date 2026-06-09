#include "kernel_operator.h"

using namespace AscendC;

class KernelTanh {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TanhCustomTilingData* tilingData)
    {
        this->tileNum = tilingData->tileNum;
        this->tileLength = tilingData->tileLength;
        this->tailLength = tilingData->tailLength;
        
        xGm.SetGlobalBuffer((__gm__ half*)x, tilingData->totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y, tilingData->totalLength);
        
        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, 1, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t length = (i == tileNum - 1) ? tailLength : tileLength;
            CopyIn(i, length);
            Compute(length);
            CopyOut(i, length);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t length)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[tileIdx * tileLength], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        Tanh(yLocal, xLocal, length);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t length)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[tileIdx * tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailLength;
};

// 修正后的标准 4 参数入口
extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, &tilingData);
    op.Process();
}
