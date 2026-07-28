#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint64_t dataPerBlock = totalLength / blockNum;
        uint64_t tail = totalLength % blockNum;
        this->blockLength = static_cast<uint32_t>(dataPerBlock + (blockIdx < tail ? 1 : 0));
        uint64_t offset = blockIdx * dataPerBlock + (blockIdx < tail ? blockIdx : tail);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_X *)z + offset, this->blockLength);

        this->tileNum = TILE_NUM;
        this->tileLength = this->blockLength / this->tileNum;
        this->tailTileLength = this->blockLength % this->tileNum;

        uint32_t maxTileLen = this->tileLength + this->tailTileLength;
        uint32_t bufferSize = maxTileLen * sizeof(DTYPE_X);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, bufferSize);
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < (int32_t)tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t offset = progress * tileLength;
        uint32_t copyLen = tileLength;
        if (progress == (int32_t)tileNum - 1 && tailTileLength > 0) {
            copyLen = tileLength + tailTileLength;
        }

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.AllocTensor<DTYPE_X>();

        AscendC::DataCopy(xLocal, xGm[offset], copyLen);
        AscendC::DataCopy(yLocal, yGm[offset], copyLen);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        uint32_t computeLen = tileLength;
        if (progress == (int32_t)tileNum - 1 && tailTileLength > 0) {
            computeLen = tileLength + tailTileLength;
        }

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.AllocTensor<DTYPE_X>();

        AscendC::Sub(zLocal, xLocal, yLocal, computeLen);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t offset = progress * tileLength;
        uint32_t copyLen = tileLength;
        if (progress == (int32_t)tileNum - 1 && tailTileLength > 0) {
            copyLen = tileLength + tailTileLength;
        }

        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.DeQue<DTYPE_X>();
        AscendC::DataCopy(zGm[offset], zLocal, copyLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm, yGm, zGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailTileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSub op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
