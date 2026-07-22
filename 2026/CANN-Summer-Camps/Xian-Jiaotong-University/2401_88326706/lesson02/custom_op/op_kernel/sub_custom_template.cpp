#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TILE_NUM = 8;

template <typename dtypeX, typename dtypeY, typename dtypeZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t baseBlockLength = totalLength / blockNum;
        uint32_t tailBlockNum = totalLength % blockNum;
        uint32_t blockOffset = blockIdx * baseBlockLength + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);

        this->blockLength = baseBlockLength + (blockIdx < tailBlockNum ? 1 : 0);
        this->tileLength = (this->blockLength + TILE_NUM * BUFFER_NUM - 1) / (TILE_NUM * BUFFER_NUM);

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, this->blockLength);

        if (this->blockLength > 0) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
            pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
        }
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; ++i) {
            uint32_t offset = i * this->tileLength;
            if (offset >= this->blockLength) {
                break;
            }
            this->currentTileLength = this->blockLength - offset;
            if (this->currentTileLength > this->tileLength) {
                this->currentTileLength = this->tileLength;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->currentTileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->currentTileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();

        AscendC::Sub(zLocal, xLocal, yLocal, this->currentTileLength);

        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();

        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->currentTileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t currentTileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tilingData, tiling);

    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
