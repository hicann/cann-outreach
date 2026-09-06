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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 1. 当前 AIV 核的编号
    int64_t blockIdx = GetBlockIdx();

    // 2. 当前核在整个 Tensor 中对应的起始元素下标
    int64_t blockOffset = blockIdx * tilingData->blockFactor;

    // 3. 计算当前核真正需要处理的元素数
    // 最后一个核可能不足 blockFactor，必须避免访问越界
    int64_t remainLength = tilingData->totalNum - blockOffset;
    blockLength_ = remainLength < tilingData->blockFactor ? remainLength : tilingData->blockFactor;

    // 一个 Tile 的元素数
    ubLength_ = tilingData->ubFactor;

    // 4. 让当前核的 GM 张量从属于自己的数据起点开始
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);

    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    // 5. 在 UB 中初始化双缓冲队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 1. 从输入队列申请一个空闲的 UB Buffer
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 2. 将当前 Tile 从 GM 搬运到 UB
    DataCopy(xLocal, inputGMX[progress], currentNum);

    // 3. 把装有数据的 Buffer 放入输入队列，等待 Compute 取走
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // 1. 从输入队列取出一个已装好输入数据的 UB Buffer
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();

    // 2. 从输出队列申请一个空闲的 UB Buffer
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 3. 在 Vector 计算单元执行 ReLU
    AscendC::Relu(yLocal, xLocal, currentNum);

    // 4. 输出结果进入输出队列，等待 CopyOut 搬回 GM
    outputQueueY.EnQue(yLocal);

    // 5. 输入 Buffer 已经使用完，释放以供后续 Tile 复用
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // 1. 从输出队列取出一个已完成 ReLU 计算的 UB Buffer
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    // 2. 将当前 Tile 从 UB 搬回当前核负责的 GM 输出区间
    DataCopy(outputGMY[progress], yLocal, currentNum);

    // 3. 输出 Buffer 已经写回 GM，可以释放并供后续 Tile 复用
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 按 ubLength_ 为步长，处理当前核负责的全部元素
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        // 最后一个 Tile 可能不足 ubLength_，需要单独处理尾块
        int64_t currentNum = blockLength_ - progress;
        currentNum = currentNum < ubLength_ ? currentNum : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
