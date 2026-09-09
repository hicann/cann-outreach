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
    // TODO 考生自行补齐
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
    // TODO 考生自行补齐
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
//   currentNum 为尾块实际元素数，DataCopyPad 可搬任意长度，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO 考生自行补齐
}

// Compute：出队 x，执行平方计算：y = x * x，结果入队
//   矢量指令要求计算长度按 32 字节对齐（float 为 8 个元素、half 为 16 个），
//   不足时向上对齐计算，多余部分仍在 buffer 范围内，CopyOut 只回拷有效部分
template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    // TODO 考生自行补齐
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO 考生自行补齐
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
//   最后一块用剩余元素数 currentNum，避免尾块越界
template <typename T>
__aicore__ inline void Square<T>::Process()
{
    // TODO 考生自行补齐
}

} // namespace NsSquare
#endif // SQUARE_H
