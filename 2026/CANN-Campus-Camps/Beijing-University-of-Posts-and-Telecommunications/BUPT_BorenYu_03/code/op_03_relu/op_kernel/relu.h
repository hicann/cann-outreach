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

constexpr int32_t BUFFER_NUM = 1;

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

    // int64_t blockLength_ = 0;
    // int64_t ubLength_ = 0;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (blockIdx < tilingData->tailBlockNum) { 
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else { 
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->tileDataNum], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const T scalar = 0.0;
    AscendC::Maxs(yLocal, xLocal, scalar, currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->tileDataNum], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t loopCount = this->tileNum;
    int64_t processDataNum = this->tileDataNum;
    for (int64_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            processDataNum = this->tailDataNum;
        }
        CopyIn(i, processDataNum);
        Compute(processDataNum);
        CopyOut(i, processDataNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
