/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
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

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    int64_t totalNum = tilingData->totalNum;
    int64_t avgBlock = tilingData->blockFactor;   // 小核基准块大小
    int64_t bigCoreNum = tilingData->bigCoreNum;  // 大核数量（前 bigCoreNum 个核）
    ubLength_ = tilingData->ubFactor;

    // "大核/小核"均匀切分：前 bigCoreNum 个核（大核）每个处理 avgBlock+1 个元素，
    // 其余核（小核）每个处理 avgBlock 个元素。任意两个核之间的负载差不超过 1 个元素，
    // 相比把余数整体堆到最后一个核上的朴素做法，能显著降低"尾核拖尾、其余核空闲等待"的问题。
    int64_t offset;
    int64_t curLen;
    if (blockIdx < bigCoreNum) {
        curLen = avgBlock + 1;
        offset = blockIdx * curLen;
    } else {
        curLen = avgBlock;
        offset = bigCoreNum * (avgBlock + 1) + (blockIdx - bigCoreNum) * avgBlock;
    }

    if (curLen <= 0 || offset >= totalNum) {
        // 当前核没有分配到数据（例如实际启动核数多于 tiling 计算出的有效核数）
        blockLength_ = 0;
        return;
    }
    // 边界保护：防止极端场景下越界读写
    if (offset + curLen > totalNum) {
        curLen = totalNum - offset;
    }
    blockLength_ = curLen;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + offset, blockLength_);

    // 双缓冲，使 CopyIn / Compute / CopyOut 三级流水可以互相掩盖
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_) * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_) * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 使用 DataCopyPad，兼容 currentNum 不是 32B 对齐倍数的场景（非整除尾块）
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // Square(x) = x * x，用 Mul(自乘) 一条向量指令完成，
    // float16 场景下底层自动按其向量算力做整块计算，性能优于逐元素标量实现
    Mul(yLocal, xLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    int64_t tailNum = blockLength_ - (loopCount - 1) * ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? tailNum : ubLength_;
        int64_t progress = i * ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
