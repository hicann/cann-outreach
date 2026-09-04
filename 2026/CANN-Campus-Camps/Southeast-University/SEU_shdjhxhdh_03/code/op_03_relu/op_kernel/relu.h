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

template <typename DT_X>
class Relu {

public:

    __aicore__ inline Relu()
    {
    }


    /*
     * =========================================================
     * Init
     * =========================================================
     */

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        /*
         * 每个核处理的数据量
         *
         * totalLength = 8 * 2048 = 16384
         *
         * blockLength = 16384 / 8 = 2048
         */

        this->blockLength =
            tilingData->totalLength /
            AscendC::GetBlockNum();


        this->tileNum =
            tilingData->tileNum;


        /*
         * 每个 tile 的元素数量
         *
         * 2048 / 16 = 128
         */

        this->tileLength =
            this->blockLength /
            this->tileNum;


        /*
         * =====================================================
         * GM 地址
         * =====================================================
         */

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x +
            this->blockLength *
            AscendC::GetBlockIdx(),
            this->blockLength
        );


        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y +
            this->blockLength *
            AscendC::GetBlockIdx(),
            this->blockLength
        );


        /*
         * =====================================================
         * DoubleBuffer
         * =====================================================
         */

        pipe.InitBuffer(
            inputQueueX,
            2,
            this->tileLength *
            sizeof(DT_X)
        );


        pipe.InitBuffer(
            outputQueueY,
            2,
            this->tileLength *
            sizeof(DT_X)
        );
    }


    /*
     * =========================================================
     * Process
     * =========================================================
     */

    __aicore__ inline void Process()
    {
        int32_t loopCount =
            this->tileNum;


        for (int32_t i = 0;
             i < loopCount;
             i++) {

            CopyIn(i);

            Compute();

            CopyOut(i);
        }
    }


private:

    /*
     * =========================================================
     * CopyIn
     * GM --> UB
     * =========================================================
     */

    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();


        DataCopy(
            xLocal,
            inputGMX[
                progress *
                this->tileLength
            ],
            this->tileLength
        );


        inputQueueX.EnQue(xLocal);
    }


    /*
     * =========================================================
     * Compute
     * =========================================================
     */

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();


        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();


        /*
         * y = max(0, x)
         */

        AscendC::Relu(
            yLocal,
            xLocal,
            this->tileLength
        );


        outputQueueY.EnQue(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }


    /*
     * =========================================================
     * CopyOut
     * UB --> GM
     * =========================================================
     */

    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();


        DataCopy(
            outputGMY[
                progress *
                this->tileLength
            ],
            yLocal,
            this->tileLength
        );


        outputQueueY.FreeTensor(yLocal);
    }


private:

    TPipe pipe;


    /*
     * DoubleBuffer
     */

    TQue<
        QuePosition::VECIN,
        2
    > inputQueueX;


    TQue<
        QuePosition::VECOUT,
        2
    > outputQueueY;


    GlobalTensor<DT_X> inputGMX;

    GlobalTensor<DT_X> outputGMY;


    uint32_t blockLength;

    uint32_t tileNum;

    uint32_t tileLength;
};


} // namespace NsRelu

#endif // RELU_H