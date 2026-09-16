#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 2048;

template <typename T>
class KernelDiv {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        totalLength = tilingData.size;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockId = GetBlockIdx();

        // 每个block处理的元素数，最后一个block处理余数
        elementsPerBlock = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        if (blockId == blockNum - 1 && remainder != 0) {
            elementsPerBlock += remainder;
        }
        offset = blockId * (totalLength / blockNum);

        // 计算tile数量和最后一个tile的长度
        tileNum = (elementsPerBlock + TILE_LENGTH - 1) / TILE_LENGTH;
        if (elementsPerBlock == 0) {
            tileNum = 0;
            return;
        }
        lastTileLength = elementsPerBlock % TILE_LENGTH;
        if (lastTileLength == 0) {
            lastTileLength = TILE_LENGTH;
        }

        xGm.SetGlobalBuffer((__gm__ T *)x + offset, elementsPerBlock);
        yGm.SetGlobalBuffer((__gm__ T *)y + offset, elementsPerBlock);
        zGm.SetGlobalBuffer((__gm__ T *)z + offset, elementsPerBlock);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t currentTileLength = (i == tileNum - 1) ? lastTileLength : TILE_LENGTH;
            CopyIn(i, currentTileLength);
            Compute(currentTileLength);
            CopyOut(i, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t tileLength)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopy(xLocal, xGm[tileIdx * TILE_LENGTH], tileLength);
        DataCopy(yLocal, yGm[tileIdx * TILE_LENGTH], tileLength);
        inQueueX.EnQue<T>(xLocal);
        inQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t tileLength)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        Div(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t tileLength)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[tileIdx * TILE_LENGTH], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    uint32_t totalLength;
    uint32_t elementsPerBlock;
    uint32_t offset;
    uint32_t tileNum;
    uint32_t lastTileLength;
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm, yGm, zGm;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    KernelDiv<half> op;
    op.Init(x, y, z, workspace, tiling);
    op.Process();
}
