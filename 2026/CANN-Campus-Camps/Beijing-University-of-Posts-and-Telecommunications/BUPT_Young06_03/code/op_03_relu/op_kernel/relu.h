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

constexpr int32_t BUFFER_NUM = 1; // 当前为双缓冲

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;

        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / tileNum;

        // 设置每个核的 Global Memory 起始地址
        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        // 为队列分配 UB 内存
        pipe.InitBuffer(
            inputQueueX,
            1,
            this->tileLength * sizeof(DT_X)
        );

        pipe.InitBuffer(
            outputQueueY,
            1,
            this->tileLength * sizeof(DT_X)
        );
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            inputGMX[progress * this->tileLength],
            this->tileLength
        );

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        // ReLU: max(x, 0)
        AscendC::Maxs(
            yLocal,
            xLocal,
            static_cast<DT_X>(0),
            this->tileLength
        );

        outputQueueY.EnQue<DT_X>(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * this->tileLength],
            yLocal,
            this->tileLength
        );

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsRelu

#endif // RELU_H