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
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    const int64_t gmOffset = blockIdx * blockFactor;
    int64_t remain = totalNum - gmOffset;
    blockLength_ = (remain > blockFactor) ? blockFactor : ((remain > 0) ? remain : 0);

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t rest = blockLength_;
    int64_t progress = 0;
    while(rest > 0){
        int64_t curr = rest > ubLength_ ? ubLength_ : rest;
        CopyIn(progress, curr);
        Compute(curr);
        CopyOut(progress, curr);
        rest -= curr;
        progress += 1;
    }
}

} // namespace NsRelu
#endif // RELU_H
