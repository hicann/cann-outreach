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

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GM_ADDR xGmAddr_;
    GM_ADDR yGmAddr_;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    blockIdx_ = AscendC::GetBlockIdx();

    // 存储 GM 地址
    xGmAddr_ = input_x;
    yGmAddr_ = output;

    // 初始化队列缓冲区（双缓冲，depth=2）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // 计算当前 block 内 GM 起始地址
    int64_t offset = blockIdx_ * blockLength_ + progress;
    GlobalTensor<T> localGMX;
    localGMX.SetGlobalBuffer((__gm__ T*)xGmAddr_ + offset, currentNum);

    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, localGMX, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xBuf = inputQueueX.DeQue<T>();
    LocalTensor<T> yBuf = outputQueueY.AllocTensor<T>();

    // Square: y = x * x
    // 使用向量指令 Mul，比 for 循环快很多
    AscendC::Mul(yBuf, xBuf, xBuf, currentNum);

    inputQueueX.FreeTensor(xBuf);
    outputQueueY.EnQue(yBuf);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // 计算当前 block 内 GM 起始地址
    int64_t offset = blockIdx_ * blockLength_ + progress;
    GlobalTensor<T> localGMY;
    localGMY.SetGlobalBuffer((__gm__ T*)yGmAddr_ + offset, currentNum);

    LocalTensor<T> yResult = outputQueueY.DeQue<T>();
    DataCopy(localGMY, yResult, currentNum);
    outputQueueY.FreeTensor(yResult);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    int64_t totalNum = blockLength_;
    int64_t loopCount = totalNum / ubLength_;
    int64_t remainder = totalNum % ubLength_;

    // DoubleBuffer: 预取下一个tile的同时计算当前tile
    // 先发起第一次CopyIn
    CopyIn(0, ubLength_);

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t progress = i * ubLength_;

        // 计算当前 tile
        Compute(ubLength_);

        // 写出当前结果
        CopyOut(progress, ubLength_);

        // 预取下一个 tile（双缓冲核心优化）
        if (i + 1 < loopCount) {
            CopyIn(progress + ubLength_, ubLength_);
        }
    }

    // 处理余数部分
    if (remainder > 0) {
        int64_t progress = loopCount * ubLength_;
        Compute(remainder);
        CopyOut(progress, remainder);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
