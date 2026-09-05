#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        uint32_t totalLength = tilingData->totalLength;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // 处理余数，确保所有数据都被处理
        uint32_t avgLength = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        this->blockLength = avgLength + (blockIdx < remainder ? 1 : 0);
        
        // 计算当前 block 的起始偏移（用三元运算符替代 min）
        uint32_t offset = blockIdx * avgLength + (blockIdx < remainder ? blockIdx : remainder);
        
        this->tileNum = tilingData->tileNum;
        
        // 处理 blockLength 为 0 的情况
        if (this->blockLength == 0) {
            this->tileNum = 0;
            this->tileLength = 0;
            return;
        }
        
        // 向上取整
        this->tileLength = (this->blockLength + tileNum - 1) / tileNum;
        
        // 重新计算实际的 tileNum，避免处理空 tile
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);
        
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        uint32_t currentOffset = progress * this->tileLength;
        uint32_t currentLength = this->tileLength;
        
        // 处理最后一个 tile 不满的情况（用三元运算符替代 min）
        if (currentOffset + currentLength > this->blockLength) {
            currentLength = this->blockLength - currentOffset;
        }
        
        this->currentTileLength = currentLength;  // 记录实际长度
        
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[currentOffset], currentLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        
        // 使用记录的实际长度
        AscendC::Relu(yLocal, xLocal, this->currentTileLength);
        
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        uint32_t currentOffset = progress * this->tileLength;
        
        DataCopy(outputGMY[currentOffset], yLocal, this->currentTileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t currentTileLength;  // 记录当前 tile 的实际长度
};

} // namespace NsRelu
#endif // RELU_H