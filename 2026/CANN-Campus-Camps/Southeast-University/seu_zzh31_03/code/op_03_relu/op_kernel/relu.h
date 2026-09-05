/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include <cstdint>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// Init：初始化全局内存张量、记录长度，并为 VECIN/VECOUT 分配双缓冲 UB
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t coreIdx = AscendC::GetBlockIdx();
    int64_t blockOffset = coreIdx * tilingData->blockFactor;
    // 本核实际负责的元素个数：最后一个核可能不足 blockFactor，需按 totalNum 截断
    int64_t remain = tilingData->totalNum - blockOffset;
    if (blockOffset >= tilingData->totalNum) {
        blockLength_ = 0;
    } else {
        blockLength_ = (tilingData->blockFactor < remain) ? tilingData->blockFactor : remain;
    }
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 仅暴露本核可见的 GM 区域，避免越界指针运算
    int64_t gmOffset = (blockLength_ > 0) ? blockOffset : 0;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    // 为输入/输出队列各分配 BUFFER_NUM 个 buffer（DoubleBuffer 机制）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// CopyIn：将当前 tile 的输入从 GM 搬运到 UB。
// 使用 DataCopyPad 可按字节精确搬运，兼容非 32B 对齐的尾部 tile。
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyPad(
        xLocal, inputGMX[progress],
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// Compute：y = max(0, x)
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// CopyOut：将计算结果从 UB 搬回 GM
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyPad(
        outputGMY[progress], yLocal, {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0});
    outputQueueY.FreeTensor(yLocal);
}

// Process：DoubleBuffer 流水，先预取下一个 tile 再计算当前 tile，提高搬运与计算的流水重叠度
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t totalTiles = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (totalTiles <= 0) {
        return;
    }

    // 预取第一个 tile
    int64_t firstNum = (ubLength_ < blockLength_) ? ubLength_ : blockLength_;
    CopyIn(0, firstNum);

    for (int64_t i = 0; i < totalTiles; i++) {
        int64_t progress = i * ubLength_;
        int64_t remain = blockLength_ - progress;
        int64_t currentNum = (ubLength_ < remain) ? ubLength_ : remain;
        // 预取下一个 tile（非最后一个）后再计算当前 tile，形成双缓冲
        if (i + 1 < totalTiles) {
            int64_t nextProgress = (i + 1) * ubLength_;
            int64_t nextRemain = blockLength_ - nextProgress;
            CopyIn(nextProgress, (ubLength_ < nextRemain) ? ubLength_ : nextRemain);
        }
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
