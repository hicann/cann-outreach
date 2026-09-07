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

    int64_t totalNum_ = 0;    // 总元素数（尾核剩余量计算用）
    int64_t blockLength_ = 0; // 每核处理的元素数（尾核已修正为有效数据量）
    int64_t ubLength_ = 0;    // 每块元素数（每次搬运/计算量）
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 每个核处理的数据量与单次搬运的块大小
    this->totalNum_ = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    // 当前核的全局内存起始偏移（多核均分，偏移按元素个数计算）
    int64_t gmOffset = this->blockLength_ * GetBlockIdx();
    // 尾核修正：totalNum 不能整除核数时，最后一个核的有效数据量不足 blockFactor，
    // 仅修正核内处理量，偏移保持不变，避免越界
    int64_t coreNum = GetBlockNum();
    if (GetBlockIdx() == coreNum - 1) {
        this->blockLength_ = this->totalNum_ - this->blockLength_ * (coreNum - 1);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, this->blockLength_);

    // 为输入/输出队列分配 UB 内存（BUFFER_NUM 个 buffer，支持 DoubleBuffer 流水）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 从全局内存搬入当前分块到输入队列（DoubleBuffer 自动切换缓冲区）
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // 取输入分块，执行 ReLU 计算（y = max(0, x)），结果送入输出队列。
    // 必须用 AscendC::Relu 显式限定：类名 Relu 在类作用域内会遮蔽该函数。
    // 仅对当前分块的有效元素（currentNum）计算，未搬入区不参与计算
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(outputLocal, inputLocal, currentNum);
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // 将输出分块从输出队列搬回全局内存（DoubleBuffer 自动切换缓冲区）
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], outputLocal, currentNum);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 单核分块循环：CopyIn -> Compute -> CopyOut，DoubleBuffer 下流水执行。
    // 每块元素数 = ubLength_，最后一块不足 ubLength_ 时按剩余量截断
    if (this->ubLength_ <= 0 || this->blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        // 当前块实际元素数（最后一块可能不足 ubLength_）
        int64_t done = i * this->ubLength_;
        int64_t remain = this->blockLength_ - done;
        int64_t currentNum = (remain < this->ubLength_) ? remain : this->ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
