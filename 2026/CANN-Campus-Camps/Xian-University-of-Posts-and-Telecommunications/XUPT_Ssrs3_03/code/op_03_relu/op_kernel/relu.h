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
    int64_t totalNum    = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    ubLength_           = tilingData->ubFactor;

    // 多核切分
    int64_t blockIdx    = GetBlockIdx();
    int64_t startOffset = blockIdx * blockFactor;

    if (startOffset >= totalNum) {
        blockLength_ = 0;
    } else {
        blockLength_ = (startOffset + blockFactor <= totalNum)
                       ? blockFactor
                       : (totalNum - startOffset);
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// ============ CopyIn：GM -> UB ============
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

// ============ Compute：优化点 1：单指令 Maxs 替代 Duplicate+Max ============
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0)，单条标量向量指令完成，比 Duplicate+Max 少一半向量计算
    Maxs<T>(yLocal, xLocal, (T)0, currentNum);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// ============ CopyOut：UB -> GM ============
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

// ============ Process：优化点 2/3：软件流水预取 + 尾块剥离 ============
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    // 整块循环次数与尾块大小（循环内不再做分支判断）
    int64_t tileNum = blockLength_ / ubLength_;
    int64_t tailNum = blockLength_ - tileNum * ubLength_;

    if (tileNum == 0) {
        // 只有一个不完整的尾块
        CopyIn(0, tailNum);
        Compute(tailNum);
        CopyOut(0, tailNum);
        return;
    }

    // 软件流水：先发起第 0 块 CopyIn
    CopyIn(0, ubLength_);

    for (int64_t i = 0; i < tileNum; ++i) {
        // 提前为下一块发起异步 CopyIn，与当前块的 Compute / CopyOut 硬件重叠
        if (i + 1 < tileNum) {
            CopyIn(i + 1, ubLength_);
        }
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }

    // 处理尾块
    if (tailNum > 0) {
        CopyIn(tileNum, tailNum);
        Compute(tailNum);
        CopyOut(tileNum, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
