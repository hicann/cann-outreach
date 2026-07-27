#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t factor)
{
    return factor == 0 ? 0 : (value + factor - 1) / factor;
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t factor)
{
    return CeilDiv(value, factor) * factor;
}

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    // Updated parameter to uint64_t
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->alignNum = 32 / sizeof(DTYPE_X);
        const uint64_t blockNum = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t totalBlockNum = CeilDiv(totalLength, this->alignNum);
        const uint64_t baseBlockNum = totalBlockNum / blockNum;
        const uint64_t tailBlockNum = totalBlockNum % blockNum;
        
        const uint64_t currentBlockNum = baseBlockNum + (blockIdx < tailBlockNum ? 1 : 0);
        const uint64_t blockOffsetInBlocks = blockIdx * baseBlockNum + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);

        this->blockLength = currentBlockNum * this->alignNum;
        this->blockOffset = blockOffsetInBlocks * this->alignNum;
        this->loopCount = TILE_NUM * BUFFER_NUM;
        
        // Ensure tileLength is also 64-bit aligned
        this->tileLength = AlignUp(CeilDiv(this->blockLength, this->loopCount), this->alignNum);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + this->blockOffset, this->blockLength);

        if (this->tileLength != 0) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
            pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->loopCount; ++i) {
            uint64_t offset = (uint64_t)i * this->tileLength;
            if (offset >= this->blockLength) {
                return;
            }
            uint64_t length = (offset + this->tileLength <= this->blockLength) ?
                this->tileLength : (this->blockLength - offset);
            CopyIn(i, length);
            Compute(i, length);
            CopyOut(i, length);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[(uint64_t)progress * this->tileLength], length);
        AscendC::DataCopy(yLocal, yGm[(uint64_t)progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress, uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Sub(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[(uint64_t)progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    
    // Updated to 64-bit
    uint64_t blockOffset;
    uint64_t blockLength;
    uint64_t tileLength;
    
    uint32_t loopCount;
    uint32_t alignNum;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}