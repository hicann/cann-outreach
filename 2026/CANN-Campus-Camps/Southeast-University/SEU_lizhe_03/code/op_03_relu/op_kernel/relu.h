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

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        this->blockLength =
            tilingData->totalLength / AscendC::GetBlockNum();

        this->tileNum = tilingData->tileNum;

        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
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
        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outQueueY.AllocTensor<DT_X>();

        AscendC::Relu(
            yLocal,
            xLocal,
            this->tileLength);

        outQueueY.EnQue(yLocal);

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outQueueY.DeQue<DT_X>();

        DataCopy(
            yGm[progress * this->tileLength],
            yLocal,
            this->tileLength);

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsRelu

#endif // RELU_H