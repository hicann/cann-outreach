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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t tileNum_ = 0;
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Init：绑定 GM 张量并初始化 UB 队列（Double Buffer）
//   blockLength_：totalNum 减去前面核已分配长度后与 blockFactor 取小，
//                 保证尾核不越界
//   ubLength_：   tiling 下发的单次 UB 循环元素数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR x, GM_ADDR y, const SquareTilingData* tilingData)
{
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    yGm_.SetGlobalBuffer((__gm__ T *)y);

    int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum > static_cast<uint64_t>(blockOffset)
        ? static_cast<int64_t>(tilingData->totalNum - blockOffset) : 0;
    blockLength_ = blockLength_ < static_cast<int64_t>(tilingData->blockFactor)
        ? blockLength_ : static_cast<int64_t>(tilingData->blockFactor);
    ubLength_ = static_cast<int64_t>(tilingData->ubFactor);
    tileNum_ = ubLength_ == 0 ? 0 : (blockLength_ + ubLength_ - 1) / ubLength_;

    xGm_.SetGlobalBuffer((__gm__ T *)x + blockOffset);
    yGm_.SetGlobalBuffer((__gm__ T *)y + blockOffset);
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, ubLength_ * sizeof(T));
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
//   currentNum 为尾块实际元素数，DataCopyPad 可搬任意长度，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    constexpr int64_t elementsPerBlock = 32 / sizeof(T);
    if (currentNum % elementsPerBlock == 0) {
        DataCopy(xLocal, xGm_[progress], currentNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[progress], copyParams, padParams);
    }
    inQueueX_.EnQue(xLocal);
}

// Compute：出队 x，执行平方计算：y = x * x，结果入队
//   矢量指令要求计算长度按 32 字节对齐（float 为 8 个元素、half 为 16 个），
//   不足时向上对齐计算，多余部分仍在 buffer 范围内，CopyOut 只回拷有效部分
template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();
    constexpr int64_t elementsPerBlock = 32 / sizeof(T);
    int64_t computeNum = ((currentNum + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    Mul(yLocal, xLocal, xLocal, computeNum);
    outQueueY_.EnQue(yLocal);
    inQueueX_.FreeTensor(xLocal);
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
    constexpr int64_t elementsPerBlock = 32 / sizeof(T);
    if (currentNum % elementsPerBlock == 0) {
        DataCopy(yGm_[progress], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[progress], yLocal, copyParams);
    }
    outQueueY_.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
//   最后一块用剩余元素数 currentNum，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::Process()
{
    for (int64_t tileIndex = 0; tileIndex < tileNum_; ++tileIndex) {
        int64_t progress = tileIndex * ubLength_;
        int64_t currentNum = (blockLength_ - progress) < ubLength_ ? (blockLength_ - progress) : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
