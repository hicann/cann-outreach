#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        uint32_t baseLength = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        this->blockLength = baseLength + (blockIdx < remainder ? 1U : 0U);
        this->blockOffset = blockIdx * baseLength + (blockIdx < remainder ? blockIdx : remainder);
        
        this->tileNum = tileNum;
        
        // ===== 修复1：使用 ceil 除法计算 tileLength =====
        uint32_t totalTiles = tileNum * BUFFER_NUM;
        this->tileLength = (this->blockLength + totalTiles - 1) / totalTiles;
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        // ===== 修复2：动态计算实际需要的循环次数 =====
        uint32_t actualLoopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        
        #pragma unroll
        for (uint32_t i = 0; i < actualLoopCount; i++) {
            // ===== 修复3：计算当前 tile 的实际长度 =====
            uint32_t currLen = this->tileLength;
            uint32_t offset = i * this->tileLength;
            if (offset + currLen > this->blockLength) {
                currLen = this->blockLength - offset;
            }
            
            CopyIn(i, currLen);
            Compute(i, currLen);
            CopyOut(i, currLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        uint32_t offset = progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        uint32_t offset = progress * this->tileLength;
        AscendC::DataCopy(zGm[offset], zLocal, len);
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
};

__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}