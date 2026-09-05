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

template <class DT_X>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData){
        auto totalLength = tilingData->totalNum;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->blockFactor;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum * BUFFER_NUM;  //循环次数是分块数*BUFFER_NUM
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress){

        AscendC::LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        AscendC::DataCopy(outputGMY[progress * this->tileLength], yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void Compute(int64_t progress){
        AscendC::LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        // AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        AscendC::Relu(yLocal, xLocal, this->tileLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    // int64_t blockLength_ = 0;
    // int64_t ubLength_ = 0;
    uint32_t blockLength; //片内总长度
    uint32_t tileNum;  //片内分块
    uint32_t tileLength;  //每块长度
};

// // TODO: 实现具体的 kernel 逻辑
// template <typename T>
// __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
// {
//     // TODO: 实现 Init 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyIn 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Compute(int64_t currentNum)
// {
//     // TODO: 实现 Compute 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyOut 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Process()
// {
//     // TODO: 实现 Process 逻辑
// }

} // namespace NsRelu
#endif // RELU_H
