/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（DoubleBuffer）

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->blockNum = tilingData->blockNum;
        this->blockFormer = tilingData->blockFormer;
        this->blockTail = tilingData->blockTail;
        this->ubFormer = tilingData->ubFormer;
        this->ubLoop = tilingData->ubLoop;
        this->ubTail = tilingData->ubTail;

        // 尾核（最后一个 block）数据量可能小于首核，循环次数与尾块大小随之变化
        uint32_t blockIdx = GetBlockIdx();
        bool isTailBlock = (blockIdx == this->blockNum - 1);
        uint64_t blockData = isTailBlock ? this->blockTail : this->blockFormer;
        this->loopCount = (blockData + this->ubFormer - 1) / this->ubFormer;
        this->tailData = blockData - (this->loopCount - 1) * this->ubFormer;
        // 每核 GM 起始偏移 = blockFormer * blockIdx
        uint64_t blockOffset = this->blockFormer * blockIdx;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockData);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockData);

        // 为 in/out 队列各分配 BUFFER_NUM 份 buffer，形成 DoubleBuffer 流水
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubFormer * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubFormer * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 尾块大小与主循环块不同，最后一次循环单独处理
        int32_t mainLoop = this->loopCount - 1;
        for (int32_t i = 0; i < mainLoop; i++) {
            CopyIn(i, this->ubFormer);
            Compute(this->ubFormer);
            CopyOut(i, this->ubFormer);
        }
        CopyIn(mainLoop, this->tailData);
        Compute(this->tailData);
        CopyOut(mainLoop, this->tailData);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint64_t dataLength)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->ubFormer], dataLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint64_t dataLength)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        // y = max(x, 0)
        AscendC::Maxs(yLocal, xLocal, (DT_X)0, dataLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint64_t dataLength)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->ubFormer], yLocal, dataLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint64_t blockNum;     // 实际使用的核数
    uint64_t blockFormer;  // 每核基础处理量（尾核除外）
    uint64_t blockTail;    // 尾核实际处理量
    uint64_t ubFormer;     // 每次 UB 搬运的元素数
    uint64_t ubLoop;       // 首核 UB 循环次数（参考值，kernel 按本核数据量重算）
    uint64_t ubTail;       // 首核尾块大小（参考值）
    uint64_t loopCount;    // 本核 UB 循环次数
    uint64_t tailData;     // 本核最后一次循环的元素数
};

} // namespace NsRelu
#endif // RELU_H
