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
    // TODO: 实现 Init 逻辑
    // 每核处理元素数、每次 UB 循环处理元素数均来自 host 侧 tiling
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // 多核切分：本核从 blockIdx * blockLength_ 处开始读写
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockLength_ * GetBlockIdx(), blockLength_);

    // 为队列分配 UB 内存（DoubleBuffer：输入/输出队列各 2 块缓冲，搬运与计算流水重叠）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    Maxs(outputLocal, inputLocal, static_cast<T>(0), currentNum);
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], outputLocal, currentNum);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t tileNum = blockLength_ / ubLength_;
    for (int64_t i = 0; i < tileNum; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    int64_t remainNum = blockLength_ % ubLength_;
    if (remainNum > 0) {
        CopyIn(tileNum, remainNum);
        Compute(remainNum);
        CopyOut(tileNum, remainNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
