/* ------------------------------------------------------------------------ */
/*  Copyright (c), Huawei Technologies Co., Ltd.                            */
/*                                                                          */
/*  Create date: 2026-01-23                                                 */
/*  Revision id: 2 (round3: 补全tanh计算逻辑)                               */
/*  Function: TanhCustom kernel 实现（多核切分 + 双buffer流水）              */
/* ------------------------------------------------------------------------ */

#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;        // 双buffer
constexpr uint32_t TILE_LENGTH = 8192;   // 每tile元素数（half下16KB/块）

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockDim, uint64_t totalSize)
    {
        uint32_t coreCount = (blockDim > 0) ? blockDim : 1;
        uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        // 每核处理一段连续数据，末核兜底余数
        uint32_t base = (uint32_t)(totalSize / coreCount);
        uint32_t offset = base * blockIdx;
        uint32_t total = (blockIdx == coreCount - 1)
                              ? (uint32_t)(totalSize - (int64_t)base * (coreCount - 1))
                              : base;
        this->totalLength = total;
        this->tileNum = (total + TILE_LENGTH - 1) / TILE_LENGTH;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset, total);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset, total);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t count = (i == tileNum - 1) ? (totalLength - TILE_LENGTH * (tileNum - 1)) : TILE_LENGTH;
            CopyIn(count);
            Compute(count);
            CopyOut(count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t count)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm, count);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::Tanh(yLocal, xLocal, count);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t count)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm, yLocal, count);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t totalLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, (uint32_t)tilingData.blockDim, tilingData.totalSize);
    op.Process();
}
