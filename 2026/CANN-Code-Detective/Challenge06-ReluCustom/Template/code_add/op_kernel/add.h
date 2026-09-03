/*!
 * \file add.h
 * \brief Add 算子 kernel 类定义
 */

#ifndef ADD_H
#define ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_tiling_data.h"
#include "add_tiling_key.h"

namespace NsAdd {

using namespace AscendC;

template <typename DT_X>
class Add {
public:
    __aicore__ inline Add() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddTilingData* tilingData)
    {
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / tileNum ;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        inputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMZ.SetGlobalBuffer((__gm__ DT_X*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, 1, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inputQueueY, 1, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueZ, 1, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
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
        LocalTensor<DT_X> yLocal = inputQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, inputGMY[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
        inputQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inputQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outputQueueZ.AllocTensor<DT_X>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outputQueueZ.EnQue<DT_X>(zLocal);
        inputQueueX.FreeTensor(xLocal);
        inputQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> zLocal = outputQueueZ.DeQue<DT_X>();
        DataCopy(outputGMZ[progress * this->tileLength], zLocal, this->tileLength);
        outputQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> inputGMY;
    GlobalTensor<DT_X> outputGMZ;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsAdd
#endif // ADD_H
