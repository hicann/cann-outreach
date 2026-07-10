#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
class KernelSubCustom {
public:
    __aicore__ inline KernelSubCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->totalLength = totalLength;
        this->tileNum = tileNum == 0 ? 1 : tileNum;

        if (blockNum == 0 || totalLength == 0) {
            this->validLength = 0;
            return;
        }

        uint64_t blockLength64 = (totalLength + blockNum - 1) / blockNum;
        uint64_t blockOffset64 = blockLength64 * blockIdx;

        if (blockOffset64 >= totalLength) {
            this->validLength = 0;
            return;
        }

        uint64_t remainLength = totalLength - blockOffset64;
        uint64_t validLength64 = remainLength > blockLength64 ? blockLength64 : remainLength;

        this->blockOffset = static_cast<uint32_t>(blockOffset64);
        this->validLength = static_cast<uint32_t>(validLength64);

        uint32_t loopCountBase = this->tileNum * BUFFER_NUM;
        this->tileLength = (this->validLength + loopCountBase - 1) / loopCountBase;
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        this->loopCount = (this->validLength + this->tileLength - 1) / this->tileLength;

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + this->blockOffset, this->validLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + this->blockOffset, this->validLength);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z *)z + this->blockOffset, this->validLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));
    }

    __aicore__ inline void Process()
    {
        if (this->validLength == 0) {
            return;
        }

        for (uint32_t i = 0; i < this->loopCount; ++i) {
            uint32_t offset = i * this->tileLength;
            this->currentLength = this->tileLength;

            if (offset + this->currentLength > this->validLength) {
                this->currentLength = this->validLength - offset;
            }

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->currentLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->currentLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

        AscendC::Sub(zLocal, xLocal, yLocal, this->currentLength);

        outQueueZ.EnQue<TYPE_Z>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();

        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->currentLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;

    uint64_t totalLength = 0;
    uint32_t blockOffset = 0;
    uint32_t validLength = 0;
    uint32_t tileNum = 1;
    uint32_t tileLength = 1;
    uint32_t loopCount = 0;
    uint32_t currentLength = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tilingData, tiling);

    KernelSubCustom<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}