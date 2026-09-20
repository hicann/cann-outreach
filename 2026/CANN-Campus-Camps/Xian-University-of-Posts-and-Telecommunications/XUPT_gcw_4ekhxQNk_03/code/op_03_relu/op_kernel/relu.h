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
    int64_t totalNum_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;  // 每个核处理的元素数量
    ubLength_ = tilingData->ubFactor;        // 单次 UB 循环的元素数量

    // 计算当前核负责的起始偏移，将 GM 绑定到各自分片，避免多核越界/覆盖
    int64_t coreOffset = GetBlockIdx() * blockLength_;
    if (coreOffset > totalNum_) {
        coreOffset = totalNum_;
    }
    int64_t sliceLen = totalNum_ - coreOffset;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreOffset, sliceLen);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreOffset, sliceLen);

    // 为双缓冲队列分配 UB 空间：输入/输出各 BUFFER_NUM 份
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // y = max(0, x)；先把 yLocal 置 0，再用 Max(x, 0) 实现 ReLU
    // （Max 不接受标量 src，因此用 Duplicate 构造一个全 0 的 LocalTensor 作 src1）
    Duplicate(yLocal, static_cast<T>(0), currentNum);
    Max(yLocal, xLocal, yLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t coreOffset = GetBlockIdx() * blockLength_;
    int64_t coreRemain = totalNum_ - coreOffset;
    if (coreRemain <= 0) {
        return;
    }
    int64_t curBlockLen = (coreRemain < blockLength_) ? coreRemain : blockLength_;

    int64_t tileNum = curBlockLen / ubLength_;
    int64_t tailNum = curBlockLen % ubLength_;

    // 主循环：双缓冲队列(BUFFER_NUM=2)自动实现 copy 与 compute 的流水线重叠
    for (int64_t i = 0; i < tileNum; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }

    // 处理尾部不足一个完整 ubFactor 的元素
    if (tailNum > 0) {
        CopyIn(tileNum, tailNum);
        Compute(tailNum);
        CopyOut(tileNum, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
