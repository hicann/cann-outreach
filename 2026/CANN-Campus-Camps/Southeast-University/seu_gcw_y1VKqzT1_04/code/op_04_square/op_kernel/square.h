/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义
 */

#ifndef ASCENDC_LESSON04_SQUARE_KERNEL_H
#define ASCENDC_LESSON04_SQUARE_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DATA_BLOCK_BYTES = 32;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY_;

    GlobalTensor<T> inputGMX_;
    GlobalTensor<T> outputGMY_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// Init：绑定 GM 张量并初始化 UB 队列（Double Buffer）
//   blockLength_：totalNum 减去前面核已分配长度后与 blockFactor 取小，
//                 保证尾核不越界
//   ubLength_：   tiling 下发的单次 UB 循环元素数
template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockOffset = tilingData->blockFactor * blockIdx;
    const int64_t remainderLength = tilingData->totalNum - blockOffset;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX_.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY_.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    pipe_.InitBuffer(inputQueueX_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(outputQueueY_, BUFFER_NUM, ubLength_ * sizeof(T));
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
//   currentNum 为尾块实际元素数，DataCopyPad 可搬任意长度，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX_.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX_[progress * ubLength_], copyParams, padParams);
    inputQueueX_.EnQue(xLocal);
}

// Compute：出队 x，执行平方计算：y = x * x，结果入队
//   矢量指令要求计算长度按 32 字节对齐（float 为 8 个元素、half 为 16 个），
//   不足时向上对齐计算，多余部分仍在 buffer 范围内，CopyOut 只回拷有效部分
template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX_.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY_.AllocTensor<T>();

    constexpr int64_t alignNum = DATA_BLOCK_BYTES / sizeof(T);
    const int64_t computeNum = ((currentNum + alignNum - 1) / alignNum) * alignNum;
    AscendC::Mul(yLocal, xLocal, xLocal, computeNum);

    outputQueueY_.EnQue<T>(yLocal);
    inputQueueX_.FreeTensor(xLocal);
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY_.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(outputGMY_[progress * ubLength_], yLocal, copyParams);
    outputQueueY_.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
//   最后一块用剩余元素数 currentNum，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSquare
#endif // ASCENDC_LESSON04_SQUARE_KERNEL_H
