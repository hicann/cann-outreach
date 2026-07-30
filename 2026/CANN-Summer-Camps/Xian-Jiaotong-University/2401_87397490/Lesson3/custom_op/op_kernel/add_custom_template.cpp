#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

namespace {
constexpr uint32_t QUEUE_DEPTH = 1;
constexpr uint32_t TILE_PER_BLOCK = 8;
static_assert(TILE_PER_BLOCK >= 1, "Tile count per block must be at least 1");
static_assert(QUEUE_DEPTH >= 1, "Queue depth must be at least 1");
}

template <typename TX, typename TY, typename TZ>
class AddCustomKernel {
public:
    __aicore__ inline AddCustomKernel() = default;

    __aicore__ inline void Init(GM_ADDR xGmAddr, GM_ADDR yGmAddr, GM_ADDR zGmAddr, uint32_t totalElemCount)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t baseBlockLen = totalElemCount / blockNum;
        const uint32_t blockTailRemain = totalElemCount % blockNum;

        if (blockIdx < blockNum - 1U) {
            blockTotalLen = baseBlockLen;
            blockStartAddr = blockIdx * baseBlockLen;
        } else {
            blockTotalLen = baseBlockLen + blockTailRemain;
            blockStartAddr = blockIdx * baseBlockLen;
        }
        baseTileLen = blockTotalLen / TILE_PER_BLOCK;
        tailTileLen = baseTileLen + (blockTotalLen % TILE_PER_BLOCK);
        xGmBuf.SetGlobalBuffer((__gm__ TX*)xGmAddr + blockStartAddr, blockTotalLen);
        yGmBuf.SetGlobalBuffer((__gm__ TY*)yGmAddr + blockStartAddr, blockTotalLen);
        zGmBuf.SetGlobalBuffer((__gm__ TZ*)zGmAddr + blockStartAddr, blockTotalLen);
        dataPipe.InitBuffer(inXQueue, QUEUE_DEPTH, baseTileLen * sizeof(TX));
        dataPipe.InitBuffer(inYQueue, QUEUE_DEPTH, baseTileLen * sizeof(TY));
        dataPipe.InitBuffer(outZQueue, QUEUE_DEPTH, baseTileLen * sizeof(TZ));
    }

    __aicore__ inline void Process()
    {
        if (blockTotalLen == 0) {
            return;
        }
        for (uint32_t i = 0; i < TILE_PER_BLOCK - 1U; ++i) {
            uint32_t offset = i * baseTileLen;
            LoadTile(offset, baseTileLen);
            ComputeTile(baseTileLen);
            StoreTile(offset, baseTileLen);
        }
        uint32_t tailOffset = (TILE_PER_BLOCK - 1U) * baseTileLen;
        LoadTile(tailOffset, tailTileLen);
        ComputeTile(tailTileLen);
        StoreTile(tailOffset, tailTileLen);
    }

private:
    __aicore__ inline void ComputeTile(uint32_t validLen)
    {
        AscendC::LocalTensor<TX> xLocal = inXQueue.DeQue<TX>();
        AscendC::LocalTensor<TY> yLocal = inYQueue.DeQue<TY>();
        AscendC::LocalTensor<TZ> zLocal = outZQueue.AllocTensor<TZ>();
        AscendC::Add(zLocal, xLocal, yLocal, validLen);
        outZQueue.EnQue<TZ>(zLocal);
        inXQueue.FreeTensor(xLocal);
        inYQueue.FreeTensor(yLocal);
    }
    __aicore__ inline void LoadTile(uint32_t gmOffset, uint32_t validLen)
    {
        AscendC::LocalTensor<TX> xLocal = inXQueue.AllocTensor<TX>();
        AscendC::LocalTensor<TY> yLocal = inYQueue.AllocTensor<TY>();
        AscendC::DataCopy(xLocal, xGmBuf[gmOffset], validLen);
        AscendC::DataCopy(yLocal, yGmBuf[gmOffset], validLen);
        inXQueue.EnQue(xLocal);
        inYQueue.EnQue(yLocal);
    }
    __aicore__ inline void StoreTile(uint32_t gmOffset, uint32_t validLen)
    {
        AscendC::LocalTensor<TZ> zLocal = outZQueue.DeQue<TZ>();
        AscendC::DataCopy(zGmBuf[gmOffset], zLocal, validLen);
        outZQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe dataPipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inXQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inYQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outZQueue;
    AscendC::GlobalTensor<TX> xGmBuf;
    AscendC::GlobalTensor<TY> yGmBuf;
    AscendC::GlobalTensor<TZ> zGmBuf;
    uint32_t blockTotalLen = 0;
    uint32_t blockStartAddr = 0;
    uint32_t baseTileLen = 0;
    uint32_t tailTileLen = 0;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tilingPara, tiling);
    AddCustomKernel<DTYPE_X, DTYPE_Y, DTYPE_Z> kernelInstance;
    kernelInstance.Init(x, y, z, tilingPara.totalLength);
    kernelInstance.Process();
}