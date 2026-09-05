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

    int64_t blockLength_; // 当前核处理的总元素数
    int64_t ubLength_;    // 单次 UB 搬运的元素数
    int64_t tileNum;      // 当前核需要循环的次数
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 1. 获取 Tiling 参数
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // 2. 计算循环次数 (向上取整)
    tileNum = (blockLength_ + ubLength_ - 1) / ubLength_;

    // 3. 设置全局内存指针
    int64_t offset = get_block_idx() * static_cast<int64_t>(blockLength_);
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    // 4. 初始化双缓冲队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto localTensor = inputQueueX.AllocTensor<T>();
    // progress * currentNum 即为全局偏移量
    DataCopy(localTensor, inputGMX[progress * currentNum], currentNum);
    inputQueueX.EnQue(localTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    auto inTensor = inputQueueX.DeQue<T>();
    auto outTensor = outputQueueY.AllocTensor<T>();

    // 执行 ReLU 计算
    // 注意：AscendC::Relu 通常不依赖 currentNum 参数，它操作整个 Tensor
    // 但如果 currentNum 小于 Tensor 大小，可能需要确保 Tensor 只分配了 currentNum 的大小
    // 在这里我们假设 AllocTensor 分配的是 ubLength_，但实际有效数据是 currentNum
    AscendC::Relu(outTensor, inTensor, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outTensor);
    inputQueueX.FreeTensor(inTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto localTensor = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * currentNum], localTensor, currentNum);
    outputQueueY.FreeTensor(localTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 1. 预取阶段：填满双缓冲
    // 注意：预取时也需计算正确的 currentNum
    for (int32_t i = 0; i < BUFFER_NUM && i < tileNum; ++i) {
        int64_t currentLen = ubLength_;
        if ((i + 1) * ubLength_ > blockLength_) {
            currentLen = blockLength_ - i * ubLength_;
        }
        CopyIn(i, currentLen);
    }

    // 2. 流水线执行阶段
    for (int32_t i = 0; i < tileNum; ++i) {
        // 计算当前批次的实际长度
        int64_t currentLen = ubLength_;
        if ((i + 1) * ubLength_ > blockLength_) {
            currentLen = blockLength_ - i * ubLength_;
        }

        Compute(currentLen);
        CopyOut(i, currentLen);
        
        // 如果还有下一批数据，提前拷贝进来
        if (i + BUFFER_NUM < tileNum) {
            int64_t nextLen = ubLength_;
            if ((i + BUFFER_NUM + 1) * ubLength_ > blockLength_) {
                nextLen = blockLength_ - (i + BUFFER_NUM) * ubLength_;
            }
            CopyIn(i + BUFFER_NUM, nextLen);
        }
    }
}

} // namespace NsRelu
#endif // RELU_H
