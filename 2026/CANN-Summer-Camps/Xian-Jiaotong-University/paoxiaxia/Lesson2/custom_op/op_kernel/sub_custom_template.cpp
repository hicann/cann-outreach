#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 4096;

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockDim = AscendC::GetBlockNum();
        this->blockLength = (totalLength + blockDim - 1) / blockDim;
        this->tileNum = tileNum;
        this->tileLength = (this->blockLength + tileNum - 1) / tileNum;

        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        uint32_t remaining = (blockOffset + this->blockLength <= totalLength)
            ? this->blockLength
            : (totalLength > blockOffset ? totalLength - blockOffset : 0);

        xGm.SetGlobalBuffer((__gm__ half*)x + blockOffset, remaining);
        yGm.SetGlobalBuffer((__gm__ half*)y + blockOffset, remaining);
        zGm.SetGlobalBuffer((__gm__ half*)z + blockOffset, remaining);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t GetTileLength(int32_t progress)
    {
        uint32_t offset = this->tileLength * progress;
        uint32_t remaining = this->blockLength - offset;
        return (remaining < this->tileLength) ? remaining : this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        uint32_t len = GetTileLength(progress);
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], len);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        uint32_t len = GetTileLength(progress);

        AscendC::Sub(zLocal, xLocal, yLocal, len);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        uint32_t len = GetTileLength(progress);
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength, tileNum, tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSub op;
    op.Init(x, y, z, tilingData.size, 8);
    op.Process();
}
