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

// 先使用稳定版本。
// 通过后如果需要再改成2做DoubleBuffer。
constexpr int32_t BUFFER_NUM = 1;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        // 每个核处理的数据量
        this->blockLength =
            tilingData->totalLength /
            AscendC::GetBlockNum();

        // 每个核切多少个tile
        this->tileNum =
            tilingData->tileNum;

        // 每次真正处理的数据长度
        this->tileLength =
            this->blockLength /
            this->tileNum /
            BUFFER_NUM;

        // 当前核对应的输入GM地址
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x +
                this->blockLength *
                    AscendC::GetBlockIdx(),
            this->blockLength);

        // 当前核对应的输出GM地址
        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y +
                this->blockLength *
                    AscendC::GetBlockIdx(),
            this->blockLength);

        // 输入Buffer
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength *
                sizeof(DT_X));

        // 输出Buffer
        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength *
                sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount =
            this->tileNum *
            BUFFER_NUM;

        for (int32_t i = 0;
             i < loopCount;
             i++) {

            CopyIn(i);

            Compute();

            CopyOut(i);
        }
    }

private:

    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        // Global Memory -> Unified Buffer
        DataCopy(
            xLocal,
            xGm[
                progress *
                this->tileLength],
            this->tileLength);

        inQueueX.EnQue(
            xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outQueueY.AllocTensor<DT_X>();

        // y = max(0, x)
        AscendC::Relu(
            yLocal,
            xLocal,
            this->tileLength);

        outQueueY.EnQue<DT_X>(
            yLocal);

        inQueueX.FreeTensor(
            xLocal);
    }

    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outQueueY.DeQue<DT_X>();

        // Unified Buffer -> Global Memory
        DataCopy(
            yGm[
                progress *
                this->tileLength],
            yLocal,
            this->tileLength);

        outQueueY.FreeTensor(
            yLocal);
    }

private:

    TPipe pipe;

    TQue<
        TPosition::VECIN,
        BUFFER_NUM>
        inQueueX;

    TQue<
        TPosition::VECOUT,
        BUFFER_NUM>
        outQueueY;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsRelu

#endif // RELU_H