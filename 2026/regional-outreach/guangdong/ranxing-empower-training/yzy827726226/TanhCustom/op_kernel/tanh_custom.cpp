/* -------------------------------------------------------------------------
 * Copyright (c) 2026 TanhCustom contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ------------------------------------------------------------------------- */

#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

#define DTYPE_X half
#define DTYPE_Y half

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim is zero");
        this->tileNum = tileNum;
        // 每核基础块长；余数交由最后一个核处理，保证所有元素都被覆盖
        uint32_t remain = totalLength % AscendC::GetBlockNum();
        this->blockLength = totalLength / AscendC::GetBlockNum();
        if (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1) {
            this->blockLength += remain;
        }
        // tile 长度向上取整，保证能覆盖本核全部元素；避免 0 长度缓冲
        this->tileLength = (this->blockLength + this->tileNum * BUFFER_NUM - 1) /
                           (this->tileNum * BUFFER_NUM);
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }
        // 实际循环次数（最后一个 tile 可能不满，由剩余长度处理）
        this->loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->loopCount; i++) {
            // 最后一个 tile 使用实际剩余长度，避免尾部元素丢失
            uint32_t curLength = this->tileLength;
            if (i == this->loopCount - 1) {
                curLength = this->blockLength - i * this->tileLength;
            }
            CopyIn(i, curLength);
            Compute(i, curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<float> tmpLocal0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmpLocal1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmpLocal2 = tmpBuf2.Get<float>();

        AscendC::Cast(tmpLocal0, xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Exp(tmpLocal1, tmpLocal0, len);
        AscendC::Muls(tmpLocal0, tmpLocal0, static_cast<float>(-1.0), len);
        AscendC::Exp(tmpLocal2, tmpLocal0, len);

        AscendC::Sub(tmpLocal0, tmpLocal1, tmpLocal2, len);
        AscendC::Add(tmpLocal1, tmpLocal1, tmpLocal2, len);
        AscendC::Div(tmpLocal2, tmpLocal0, tmpLocal1, len);

        AscendC::Cast(yLocal, tmpLocal2, AscendC::RoundMode::CAST_ROUND, len);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    int32_t loopCount;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
