#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 256;

__aicore__ inline uint32_t CeilDiv(uint64_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}
}

class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t baseBlockLength = totalLength / blockNum;
        const uint32_t tailBlockCount = totalLength % blockNum;
        const uint32_t extra = (blockIdx < tailBlockCount) ? 1U : 0U;
        blockLength_ = baseBlockLength + extra;
        const uint64_t blockOffset =
            blockIdx * baseBlockLength + (blockIdx < tailBlockCount ? blockIdx : tailBlockCount);

        tileNum_ = CeilDiv(blockLength_, TILE_LENGTH);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockOffset, blockLength_);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y + blockOffset, blockLength_);
        zGm.SetGlobalBuffer((__gm__ DTYPE_X *)z + blockOffset, blockLength_);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        for (uint32_t tileIdx = 0; tileIdx < tileNum_; ++tileIdx) {
            const uint32_t currentTileLength = GetCurrentTileLength(tileIdx);
            CopyIn(tileIdx, currentTileLength);
            Compute(currentTileLength);
            CopyOut(tileIdx, currentTileLength);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentTileLength(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * TILE_LENGTH;
        const uint32_t remain = blockLength_ - offset;
        return remain < TILE_LENGTH ? remain : TILE_LENGTH;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.AllocTensor<DTYPE_X>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        AscendC::DataCopy(xLocal, xGm[offset], currentTileLength);
        AscendC::DataCopy(yLocal, yGm[offset], currentTileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.AllocTensor<DTYPE_X>();
        AscendC::Sub(zLocal, xLocal, yLocal, currentTileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.DeQue<DTYPE_X>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        AscendC::DataCopy(zGm[offset], zLocal, currentTileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_X> yGm;
    AscendC::GlobalTensor<DTYPE_X> zGm;
    uint64_t blockLength_ = 0;
    uint32_t tileNum_ = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSub op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
