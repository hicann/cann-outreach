#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() : isValid(false) {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                 uint64_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // ✅ 修复 P1: 处理不能整除的情况，避免尾部元素丢失
        this->totalLength = totalLength;
        this->blockNum = blockNum;
        this->blockIdx = blockIdx;
        
        // 计算每个核的基础数据量
        const uint32_t baseLen = totalLength / blockNum;
        const uint32_t remainder = totalLength % blockNum;
        
        // 前 remainder 个核多处理 1 个元素
        const uint32_t extra = (blockIdx < remainder) ? 1 : 0;
        const uint32_t startOffset = blockIdx * baseLen + 
                                      ((blockIdx < remainder) ? blockIdx : remainder);
        this->blockLength = baseLen + extra;
        
        // 如果当前核没有数据，标记为无效
        if (this->blockLength == 0) {
            this->isValid = false;
            return;
        }
        
        // ✅ 修复 P2: 使用传入的 tileNum，而不是硬编码
        this->tileNum = tileNum;
        
        // 计算 tileLength，确保能处理边界情况
        // 使用 (blockLength + tileNum * BUFFER_NUM - 1) / (tileNum * BUFFER_NUM) 向上取整
        const uint32_t totalTiles = tileNum * BUFFER_NUM;
        this->tileLength = (this->blockLength + totalTiles - 1) / totalTiles;
        
        // 如果 tileLength 为 0，说明数据量太小，直接处理全部
        if (this->tileLength == 0) {
            this->tileLength = this->blockLength;
            this->tileNum = 1;
        }
        
        // 设置全局内存指针
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + startOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + startOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + startOffset, this->blockLength);
        
        // 初始化队列缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
        
        this->isValid = true;
    }

    __aicore__ inline void Process()
    {
        if (!this->isValid) {
            return;
        }

        // 计算实际需要的循环次数（处理边界情况）
        const uint32_t totalTiles = (this->blockLength + this->tileLength - 1) / this->tileLength;
        const uint32_t loopCount = totalTiles;
        
        if (loopCount == 0) {
            return;
        }
        
        // ✅ 修复 P1: 使用动态长度，最后一个 tile 可能更小
        for (uint32_t i = 0; i < loopCount; i++) {
            // 计算当前 tile 的实际长度（最后一个可能不足）
            const uint32_t currentLen = (i == loopCount - 1) ?
                                         this->blockLength - i * this->tileLength :
                                         this->tileLength;
            
            CopyIn(i, currentLen);
            Compute(i, currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], len);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Sub(zLocal, xLocal, yLocal, len);  // ✅ 使用实际长度
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, len);  // ✅ 使用实际长度
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
    uint64_t totalLength = 0;
    uint32_t blockNum = 0;
    uint32_t blockIdx = 0;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
    bool isValid = false;
};

__global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tiling_data, tiling);
    // ✅ 修复 P2: 使用 tiling 数据中的 tileNum，而不是硬编码
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}