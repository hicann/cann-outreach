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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每个核处理的数据长度（多核并行切分）
        this->blockLength = static_cast<int32_t>(tilingData->totalLength / AscendC::GetBlockNum());
        this->tileNum = static_cast<int32_t>(tilingData->tileNum);
        // 单块处理的元素数（性能优化：每块尽量大，循环次数尽量少）
        this->tileLength = this->blockLength / this->tileNum;
        // 设置每个核的 Global Memory 起始地址（多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存（双缓冲：input / output 各 BUFFER_NUM 块）
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 双缓冲流水：先预取第 0 块，之后每轮 计算当前块 → 预取下一块 → 搬出当前块，
        // 使下一块的 DMA 搬运与当前块的计算/搬出重叠，隐藏搬运延迟。
        int32_t loopCount = this->tileNum;
        CopyIn(0);
        for (int32_t i = 0; i < loopCount; i++) {
            Compute();
            if (i + 1 < loopCount) {
                CopyIn(i + 1); // 预取下一块（双缓冲，不覆盖正在计算的 buffer）
            }
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        // y = max(0, x)，使用 AscendC 向量指令 Relu
        AscendC::Relu(yLocal, xLocal, this->tileLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    int32_t blockLength; // 每个核处理的元素数
    int32_t tileNum;     // 单核内分块数（每块 tileLength 个元素）
    int32_t tileLength;  // 单块处理的元素数
};

} // namespace NsRelu
#endif // RELU_H

