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

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t actualNum_ = 0; // 本核实际处理的元素数（尾核可能不足 blockLength_）
    int64_t tileNum_ = 0;   // 本核 UB 循环次数
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    // 每核处理的元素数、每次 UB 循环处理的元素数
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;
    // 本核在 GM 上的起始偏移（多核切分）
    int64_t start = this->blockLength_ * AscendC::GetBlockIdx();
    // 尾核数据可能不足一个 blockLength_, 按剩余量截断
    int64_t remain = tilingData->totalNum - start;
    this->actualNum_ = remain < this->blockLength_ ? remain : this->blockLength_;
    this->tileNum_ = (this->actualNum_ + this->ubLength_ - 1) / this->ubLength_;
    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + start, this->actualNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + start, this->actualNum_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum * static_cast<int64_t>(sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    } else {
        // N 非 32 倍数时尾块不足 32B 对齐, 用 DataCopyPad 搬运
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * this->ubLength_], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Mul(yLocal, xLocal, xLocal, currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum * static_cast<int64_t>(sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * this->ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    for (int64_t i = 0; i < this->tileNum_; i++) {
        // 最后一次循环可能不足 ubLength_
        int64_t remain = this->actualNum_ - i * this->ubLength_;
        int64_t currentNum = remain < this->ubLength_ ? remain : this->ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
