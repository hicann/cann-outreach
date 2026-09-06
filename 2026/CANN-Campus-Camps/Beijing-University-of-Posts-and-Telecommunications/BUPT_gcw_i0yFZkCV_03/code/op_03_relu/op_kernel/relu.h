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

// 设备侧整除向上取整（int64_t）
__aicore__ inline int64_t ReluCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

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

    int64_t blockLength_ = 0;   // 本核实际处理的元素数（尾核可能小于 blockFactor）
    int64_t ubLength_ = 0;      // 每轮进 UB 的元素数
    int64_t totalNum_ = 0;      // 数据总长度
    int64_t roundCount_ = 0;    // 循环轮数 = ceil(blockLength_ / ubLength_)，空闲核为 0
};

// Init：解析 tiling + 绑定本核 GM 窗口 + 申报 UB 缓冲
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;

    // 核间窗口：本核负责 [blockIdx * blockFactor, +blockFactor)
    int64_t startOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (startOffset >= totalNum_) {
        // 数据已被前面的核覆盖完，本核空闲
        roundCount_ = 0;
        return;
    }
    // 尾核实际处理量可能小于 blockFactor
    int64_t remain = totalNum_ - startOffset;
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;

    inputGMX.SetGlobalBuffer((__gm__ T *)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + startOffset, blockLength_);

    // UB 切分方案：输入/输出队列各 BUFFER_NUM 份缓冲（双缓冲：
    // 第 i+1 轮 MTE 搬入与第 i 轮 Maxs 计算重叠，掩盖搬运延迟）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    roundCount_ = ReluCeilDiv(blockLength_, ubLength_);
}

// CopyIn：MTE2 异步搬入第 progress 轮（currentNum 个元素）到 UB
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

// Compute：y = max(x, 0)，Vector 单元 128-lane 指令按 repeat 展开
// 使用 AscendC 专用高阶 API Relu(dst, src, count)，对 half/float 两精度均有重载
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// CopyOut：MTE3 异步搬出第 progress 轮结果到 GM
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// Process：流水线主循环。前若干轮满 ubLength_，尾轮按剩余量（对齐有保证：
// blockFactor 与 ubFactor 均为 32B 元素数的倍数，尾轮 currentNum 亦然）
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (roundCount_ == 0) {
        return;
    }
    for (int64_t i = 0; i < roundCount_; i++) {
        int64_t currentNum = (i == roundCount_ - 1)
                                 ? (blockLength_ - (roundCount_ - 1) * ubLength_)
                                 : ubLength_;
        CopyIn(i, currentNum);    // MTE2：HBM → UB
        Compute(currentNum);      // Vector：UB 内 max(x, 0)
        CopyOut(i, currentNum);   // MTE3：UB → HBM
    }
}

} // namespace NsRelu
#endif // RELU_H
