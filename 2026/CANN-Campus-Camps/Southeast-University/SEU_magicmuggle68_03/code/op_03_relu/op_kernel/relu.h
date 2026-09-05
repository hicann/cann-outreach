/*!
 * \file relu.h
 * \brief Relu 算子八核大块处理
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr uint8_t BUFFER_NUM = 1;
constexpr uint8_t QUEUE_DEPTH = 1;

template <typename T, bool USE_EXPLICIT_RELU>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        blockLength_ =
            tilingData->blockFactor;

        const int64_t blockOffset =
            static_cast<int64_t>(GetBlockIdx()) *
            blockLength_;

        inputGMX.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x) +
                blockOffset,
            blockLength_);

        outputGMY.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y) +
                blockOffset,
            blockLength_);

        const uint32_t bufferBytes =
            static_cast<uint32_t>(
                blockLength_ * sizeof(T));

        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            bufferBytes);

        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            bufferBytes);
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> inputLocal =
            inputQueueX.AllocTensor<T>();

        DataCopy(
            inputLocal,
            inputGMX,
            blockLength_);

        inputQueueX.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> inputLocal =
            inputQueueX.DeQue<T>();

        LocalTensor<T> outputLocal =
            outputQueueY.AllocTensor<T>();

        if constexpr (USE_EXPLICIT_RELU) {
            constexpr uint64_t elementsPerRepeat =
                256 / sizeof(T);

            const uint8_t repeatTimes =
                static_cast<uint8_t>(
                    blockLength_ /
                    elementsPerRepeat);

            AscendC::Relu<T, true>(
                outputLocal,
                inputLocal,
                elementsPerRepeat,
                repeatTimes,
                {1, 1, 8, 8});
        } else {
            AscendC::Relu(
                outputLocal,
                inputLocal,
                blockLength_);
        }

        outputQueueY.EnQue(outputLocal);
        inputQueueX.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> outputLocal =
            outputQueueY.DeQue<T>();

        DataCopy(
            outputGMY,
            outputLocal,
            blockLength_);

        outputQueueY.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, QUEUE_DEPTH>
        inputQueueX;

    TQue<QuePosition::VECOUT, QUEUE_DEPTH>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
};

} // namespace NsRelu

#endif // RELU_H