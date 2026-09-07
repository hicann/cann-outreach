// Kernel-side implementation.
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t blockLength,
        uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t offset = blockIdx * blockLength;

        this->blockLength = 0;
        if (offset < totalLength) {
            this->blockLength = totalLength - offset;
            if (this->blockLength > blockLength) {
                this->blockLength = blockLength;
            }
        }

        this->tileLength = tileLength;
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t dataCount = GetTileDataCount(i);
            CopyIn(i, dataCount);
            Compute(dataCount);
            CopyOut(i, dataCount);
        }
    }

private:
    __aicore__ inline uint32_t GetTileDataCount(uint32_t progress) const
    {
        uint32_t offset = progress * this->tileLength;
        uint32_t remain = this->blockLength - offset;
        return remain > this->tileLength ? this->tileLength : remain;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t dataCount)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        uint32_t offset = progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], dataCount);
        AscendC::DataCopy(yLocal, yGm[offset], dataCount);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t dataCount)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, dataCount);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t dataCount)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        uint32_t offset = progress * this->tileLength;
        AscendC::DataCopy(zGm[offset], zLocal, dataCount);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);
    op.Process();
}
