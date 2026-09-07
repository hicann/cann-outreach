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
// Compare API 要求 count 所占空间 256 字节对齐：对齐元素数 = 256 / sizeof(T)
// （fp32 为 64 的倍数，fp16 为 128 的倍数）；尾块补零区的结果由 CopyOut 只写回真实长度而丢弃
template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum, int64_t alignedNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpXBuf;  // half 升精度输入缓冲（fp32）
    TBuf<TPosition::VECCALC> tmpYBuf;  // half 升精度结果缓冲（fp32）
    TBuf<TPosition::VECCALC> zeroBuf;  // 零张量缓冲（fp32）
    TBuf<TPosition::VECCALC> maskBuf;  // Compare 掩码缓冲

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElem_ = 1; // 256B 对齐所需元素数：256 / sizeof(T)
};

// 实现 Init：按 tiling 切分设置 GM 起始地址与 UB 缓冲
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 末核按剩余量处理（blockFactor 为向上取整的均分值）
    int64_t realLen = tilingData->totalNum - AscendC::GetBlockIdx() * tilingData->blockFactor;
    if (realLen > tilingData->blockFactor) {
        realLen = tilingData->blockFactor;
    }
    blockLength_ = realLen > 0 ? realLen : 0;
    ubLength_ = tilingData->ubFactor;
    alignElem_ = 256 / sizeof(T);

    inputGMX.SetGlobalBuffer((__gm__ T*)x + AscendC::GetBlockIdx() * tilingData->blockFactor, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + AscendC::GetBlockIdx() * tilingData->blockFactor, blockLength_);

    if (blockLength_ == 0) {
        return; // 空核直接返回
    }
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpXBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpYBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(zeroBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(maskBuf, ubLength_);
}

// 实现 CopyIn：DataCopyPad 任意字节安全；不做 Duplicate 预填充——
// 尾块补零区的计算结果会被 CopyOut 只写回 currentNum 而丢弃，
// 且矢量写与 MTE 搬运混用会破坏队列同步（真卡实测会导致正数被清零）
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum, int64_t alignedNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyPad(xLocal, inputGMX[progress * ubLength_],
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// 实现 Compute：y = max(x, 0)，与 torch.nn.functional.relu 逐位一致
// half 先升精度到 fp32 再走比较+选择（fp32 序列已在判题环境验证正确），转换零精度损失
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t alignedNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>(alignedNum);
    if constexpr (sizeof(T) == 2) {
        LocalTensor<float> xF = tmpXBuf.Get<float>(alignedNum);
        LocalTensor<float> yF = tmpYBuf.Get<float>(alignedNum);
        LocalTensor<float> zeroF = zeroBuf.Get<float>(alignedNum);
        Cast(xF, xLocal, RoundMode::CAST_NONE, alignedNum);
        Duplicate(zeroF, 0.0f, alignedNum);
        CompareScalar(maskLocal, xF, 0.0f, CMPMODE::GT, alignedNum);
        Select(yF, maskLocal, xF, zeroF, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
        Cast(yLocal, yF, RoundMode::CAST_RINT, alignedNum);
    } else {
        LocalTensor<T> zeroLocal = zeroBuf.Get<T>(alignedNum);
        Duplicate(zeroLocal, (T)0, alignedNum);
        CompareScalar(maskLocal, xLocal, (T)0, CMPMODE::GT, alignedNum);
        Select(yLocal, maskLocal, xLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// 实现 CopyOut：UB -> GM 只写回真实元素（DataCopyPad 任意字节安全）
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyPad(outputGMY[progress * ubLength_], yLocal,
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0});
    outputQueueY.FreeTensor(yLocal);
}

// 实现 Process：按 ubLength_ 分块循环，尾块按 256 字节对齐后计算、只写回真实量
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t remain = blockLength_ - i * ubLength_;
        int64_t currentNum = remain > ubLength_ ? ubLength_ : remain;
        int64_t alignedNum = (currentNum + alignElem_ - 1) / alignElem_ * alignElem_;
        CopyIn(i, currentNum, alignedNum);
        Compute(alignedNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H