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

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
        const int64_t remaining = tilingData->totalNum > offset ? tilingData->totalNum - offset : 0;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        tileLength = static_cast<uint32_t>(tilingData->ubFactor);
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength;) {
            const int64_t remaining = blockLength - offset;
            currentNum = remaining < tileLength ? static_cast<uint32_t>(remaining) : tileLength;
            CopyIn(offset);
            Compute();
            CopyOut(offset);
            offset += currentNum;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        if ((currentNum * sizeof(DT_X)) % 32 == 0) {
            DataCopy(xLocal, inputGMX[offset], currentNum);
        } else {
            DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, currentNum);
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        if ((currentNum * sizeof(DT_X)) % 32 == 0) {
            DataCopy(outputGMY[offset], yLocal, currentNum);
        } else {
            DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    int64_t blockLength;
    uint32_t tileLength;
    uint32_t currentNum;
};

} // namespace NsRelu
#endif // RELU_H
