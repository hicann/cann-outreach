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

// ============ Init：解析 tiling、绑定 GM、分配 UB Buffer ============
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 1. 解析 Tiling 参数
    int64_t totalNum    = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    ubLength_           = tilingData->ubFactor;

    // 2. 多核切分：计算当前核的 GM 起始偏移与处理长度
    int64_t blockIdx    = GetBlockIdx();
    int64_t startOffset = blockIdx * blockFactor;

    if (startOffset >= totalNum) {
        blockLength_ = 0;   // 尾核空载保护
    } else {
        blockLength_ = (startOffset + blockFactor <= totalNum)
                       ? blockFactor
                       : (totalNum - startOffset);
    }

    // 3. 绑定 Global Memory 起始地址与长度
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // 4. 为输入/输出 Queue 分配 DoubleBuffer（各 BUFFER_NUM 份）
    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// ============ CopyIn：GM -> UB 搬运 ============
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 1. 从 VECIN Queue 申请一块 LocalTensor
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 2. GM -> UB 异步搬运（按当前 tile 的实际长度）
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);

    // 3. 入队（触发 MTE 事件同步）
    inputQueueX.EnQue(xLocal);
}

// ============ Compute：y = max(x, 0) ============
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // 1. 从输入 Queue 取出 Tensor
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();

    // 2. 从输出 Queue 申请一块结果 Tensor
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 3. ReLU 计算：先用 Duplicate 填 0，再用 Max 取逐元素最大值
    Duplicate<T>(yLocal, (T)0, currentNum);
    Max<T>(yLocal, xLocal, yLocal, currentNum);

    // 4. 结果入队
    outputQueueY.EnQue<T>(yLocal);

    // 5. 释放输入 Tensor（归还 VECIN Queue，避免双缓冲卡死）
    inputQueueX.FreeTensor(xLocal);
}

// ============ CopyOut：UB -> GM 搬运 ============
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // 1. 从 VECOUT Queue 取出结果 Tensor
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    // 2. UB -> GM 搬运（按当前 tile 的实际长度）
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);

    // 3. 释放 Tensor
    outputQueueY.FreeTensor(yLocal);
}

// ============ Process：主循环，多核 + UB 双重切分 ============
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 当前核无任务，直接返回（尾核空载保护）
    if (blockLength_ <= 0) {
        return;
    }

    // 当前核需要循环的 tile 次数
    int64_t tileNum = (blockLength_ + ubLength_ - 1) / ubLength_;

    // DoubleBuffer 流水线主循环
    for (int64_t i = 0; i < tileNum; ++i) {
        // 计算当前 tile 的实际元素数（处理尾块）
        int64_t remaining  = blockLength_ - i * ubLength_;
        int64_t currentNum = (ubLength_ < remaining) ? ubLength_ : remaining;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H