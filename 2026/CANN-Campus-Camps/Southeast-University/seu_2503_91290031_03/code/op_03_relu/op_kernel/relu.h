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
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    int64_t tileNum_ = 0;
    int64_t startOffset_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    int64_t totalNum = tilingData->totalNum;
    int64_t baseLength = totalNum / blockNum;
    int64_t remainder = totalNum % blockNum;

    // 单核时 remainder = 0，blockNum = 1
    this->startOffset_ = blockIdx * baseLength + (blockIdx < remainder ? blockIdx : remainder);
    this->blockLength_ = baseLength + (blockIdx < remainder ? 1 : 0);
    
    // 防止 blockLength_ 为 0
    if (this->blockLength_ < 1) {
        this->blockLength_ = 1;
    }
    
    this->tileLength_ = tilingData->blockFactor;
    if (this->tileLength_ < 1) {
        this->tileLength_ = 1;
    }
    if (this->tileLength_ > this->blockLength_) {
        this->tileLength_ = this->blockLength_;
    }
    
    this->tileNum_ = (this->blockLength_ + this->tileLength_ - 1) / this->tileLength_;
    if (this->tileNum_ < 1) {
        this->tileNum_ = 1;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + this->startOffset_, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + this->startOffset_, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int32_t progress)
{
    int64_t offset = progress * this->tileLength_;
    int64_t copySize = this->tileLength_;
    if (offset + copySize > this->blockLength_) {
        copySize = this->blockLength_ - offset;
    }
    if (copySize <= 0) {
        return;
    }

    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::Duplicate(xLocal, (T)0, this->tileLength_);
    DataCopy(xLocal, inputGMX[offset], copySize);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute()
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    AscendC::Duplicate(yLocal, (T)0, this->tileLength_);

    LocalTensor<T> zeroLocal = outputQueueY.AllocTensor<T>();
    AscendC::Duplicate(zeroLocal, (T)0, this->tileLength_);
    AscendC::Max(yLocal, xLocal, zeroLocal, this->tileLength_);
    outputQueueY.FreeTensor(zeroLocal);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int32_t progress)
{
    int64_t offset = progress * this->tileLength_;
    int64_t copySize = this->tileLength_;
    if (offset + copySize > this->blockLength_) {
        copySize = this->blockLength_ - offset;
    }
    if (copySize <= 0) {
        return;
    }

    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[offset], yLocal, copySize);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int32_t i = 0; i < this->tileNum_; i++) {
        CopyIn(i);
        Compute();
        CopyOut(i);
    }
}

}  // namespace NsRelu
#endif  // RELU_H