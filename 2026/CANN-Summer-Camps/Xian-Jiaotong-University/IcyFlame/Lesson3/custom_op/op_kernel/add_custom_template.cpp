#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t baseBlockLength = totalLength / blockNum;
        const uint64_t remainder = totalLength % blockNum;
        this->blockLength = static_cast<uint32_t>(baseBlockLength + (blockIdx < remainder ? 1 : 0));
        const uint64_t blockOffset = blockIdx * baseBlockLength + (blockIdx < remainder ? blockIdx : remainder);
        this->tileNum = tileNum;
        const uint32_t loopCount = tileNum * BUFFER_NUM;
        this->tileLength = (this->blockLength + loopCount - 1) / loopCount;

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, this->blockLength);

        if (this->blockLength == 0) {
            return;
        }
        const auto xBufferSize = (this->tileLength * sizeof(dtypeX) + 31) / 32 * 32;
        const auto yBufferSize = (this->tileLength * sizeof(dtypeY) + 31) / 32 * 32;
        const auto zBufferSize = (this->tileLength * sizeof(dtypeZ) + 31) / 32 * 32;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, xBufferSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, yBufferSize);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, zBufferSize);
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            const uint32_t offset = i * this->tileLength;
            if (offset >= this->blockLength) {
                break;
            }
            const uint32_t remainLength = this->blockLength - offset;
            const uint32_t currentLength = remainLength < this->tileLength ? remainLength : this->tileLength;
            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[offset], length);
        AscendC::DataCopy(yLocal, yGm[offset], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[offset], zLocal, length);
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
    uint32_t tileNum;
    uint32_t tileLength;
};

 __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
