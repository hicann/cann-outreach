#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#ifndef DTYPE_Z
#define DTYPE_Z half
#endif

constexpr int32_t BUFFER_NUM = 2;

template <class TX, class TY, class TZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() : blockLength(0), tileNum(0), tileLength(0), lastTileLen(0) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t numCores = AscendC::GetBlockNum();
        uint32_t coreIdx  = AscendC::GetBlockIdx();
        uint32_t baseLen  = totalLength / numCores;
        uint32_t remainder = totalLength % numCores;
        this->blockLength = baseLen + (coreIdx < remainder ? 1 : 0);
        uint32_t offset   = coreIdx * baseLen + (coreIdx < remainder ? coreIdx : remainder);

        this->tileNum = tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        uint32_t totalTiles = this->tileNum * BUFFER_NUM;
        this->lastTileLen = this->blockLength - this->tileLength * (totalTiles - 1);

        xGm.SetGlobalBuffer((__gm__ TX *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ TY *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TZ *)z + offset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->lastTileLen * sizeof(TX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->lastTileLen * sizeof(TY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->lastTileLen * sizeof(TZ));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? this->lastTileLen : this->tileLength;
            CopyIn(i, curLen);
            Compute(i, curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<TX> xLocal = inQueueX.AllocTensor<TX>();
        AscendC::LocalTensor<TY> yLocal = inQueueY.AllocTensor<TY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        (void)progress;
        AscendC::LocalTensor<TX> xLocal = inQueueX.DeQue<TX>();
        AscendC::LocalTensor<TY> yLocal = inQueueY.DeQue<TY>();
        AscendC::LocalTensor<TZ> zLocal = outQueueZ.AllocTensor<TZ>();
        AscendC::Sub(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<TZ> zLocal = outQueueZ.DeQue<TZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TX> xGm;
    AscendC::GlobalTensor<TY> yGm;
    AscendC::GlobalTensor<TZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLen;
};

__global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tiling_data, tiling);
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
