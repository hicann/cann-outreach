#include "kernel_operator.h"
#include "add_custom_template_tiling.h"
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t baseLength = totalLength / blockNum;
        const uint32_t remain = totalLength % blockNum;
        this->blockLength = baseLength + (blockIdx < remain ? 1U : 0U);
        this->blockOffset = blockIdx * baseLength + (blockIdx < remain ? blockIdx : remain);
        this->tileNum = tileNum;
        const uint32_t loopCount = tileNum * BUFFER_NUM;
        this->tileLength = this->blockLength == 0 ? 1 : (this->blockLength + loopCount - 1) / loopCount;
        this->loopCount = this->blockLength == 0 ? 0 : (this->blockLength + this->tileLength - 1) / this->tileLength;
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockOffset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->loopCount; ++i) {
            const uint32_t currentLength = GetCurrentLength(i);
            CopyIn(i, currentLength);
            Compute(currentLength);
            CopyOut(i, currentLength);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentLength(uint32_t progress) const
    {
        const uint32_t offset = progress * this->tileLength;
        const uint32_t left = this->blockLength - offset;
        return left < this->tileLength ? left : this->tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], currentLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], currentLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, currentLength);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, currentLength);
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
    uint32_t blockOffset;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t loopCount;
};

 __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    (void)workspace;
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
