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

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->blockFactor;
        // 单次处理的元素数（每块长度）
        this->tileLength = tilingData->ubFactor;
        // 本核在 Global Memory 上的起始偏移
        int64_t start = static_cast<int64_t>(this->blockLength) * AscendC::GetBlockIdx();
        // 本核实际处理的元素数（最后一核可能不足 blockLength）
        int64_t remain = tilingData->totalNum - start;
        this->actualLength = remain < this->blockLength ? static_cast<uint32_t>(remain) : this->blockLength;
        // 单核内分块数（向上取整，最后一块可能不足 tileLength）
        this->tileNum = (this->actualLength + this->tileLength - 1) / this->tileLength;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + start, this->actualLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + start, this->actualLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 第 progress 块实际处理的元素数（最后一块可能不足 tileLength）
    __aicore__ inline uint32_t GetTileLength(uint32_t progress)
    {
        uint32_t remain = this->actualLength - progress * this->tileLength;
        return remain < this->tileLength ? remain : this->tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        uint32_t copyLength = GetTileLength(progress);
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        if (copyLength * sizeof(DT_X) % 32 == 0) {
            DataCopy(xLocal, inputGMX[progress * this->tileLength], copyLength);
        } else {
            // 尾块不足 32B 对齐时, 用 DataCopyPad 搬运(不对齐部分自动处理)
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, inputGMX[progress * this->tileLength], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, GetTileLength(progress));
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        uint32_t copyLength = GetTileLength(progress);
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        if (copyLength * sizeof(DT_X) % 32 == 0) {
            DataCopy(outputGMY[progress * this->tileLength], yLocal, copyLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[progress * this->tileLength], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;  // 每核处理的元素数
    uint32_t tileNum;      // 单核内分块数
    uint32_t tileLength;   // 单核内每块元素数
    uint32_t actualLength; // 本核实际处理的元素数
};

} // namespace NsRelu
#endif // RELU_H
