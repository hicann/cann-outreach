/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atanh.h
 * \brief Atanh 算子 kernel 类定义
 *
 * 计算公式: atanh(x) = 0.5 * ln((1 + x) / (1 - x)), x ∈ (-1, 1)
 *
 * 实现步骤:
 *   1. CopyIn: x (GM) -> UB
 *   2. Compute:
 *      a. tmp1 = 1.0 + x
 *      b. tmp2 = 1.0 - x
 *      c. tmp3 = tmp1 / tmp2     => (1+x)/(1-x)
 *      d. tmp3 = Ln(tmp3)        => ln((1+x)/(1-x))
 *      e. tmp3 = tmp3 * 0.5      => 0.5 * ln(...) = atanh(x)
 *   3. CopyOut: result (UB) -> y (GM)
 *
 * 模板参数说明:
 *   - T: 数据类型（half/float16）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 */

#ifndef ATANH_H
#define ATANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atanh_tiling_data.h"
#include "atanh_tiling_key.h"

namespace NsAtanh {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class Atanh {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Atanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AtanhTilingData* tilingData);
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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const AtanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();

    // 分配临时计算缓冲区
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpOnesBuf;
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpOnesBuf, ubLength_ * sizeof(T));

    AscendC::LocalTensor<T> tmp1 = tmpBuf1.Get<T>();      // 用于 (1+x)
    AscendC::LocalTensor<T> tmp2 = tmpBuf2.Get<T>();      // 用于 (1-x) -> (1+x)/(1-x) -> ln -> *0.5 -> final
    AscendC::LocalTensor<T> tmpOnes = tmpOnesBuf.Get<T>(); // 常量 1.0

    // 1. tmpOnes = 1.0
    AscendC::Duplicate(tmpOnes, (T)1.0, currentNum);

    // 2. tmp1 = 1.0 + x
    AscendC::Add(tmp1, xLocal, tmpOnes, currentNum);

    // 3. tmp2 = 1.0 - x
    AscendC::Sub(tmp2, tmpOnes, xLocal, currentNum);

    // 4. tmp2 = tmp1 / tmp2  => (1+x)/(1-x)
    AscendC::Div(tmp2, tmp1, tmp2, currentNum);

    // 5. tmp2 = Ln(tmp2)  => ln((1+x)/(1-x))
    AscendC::Ln(tmp2, tmp2, currentNum);

    // 6. tmp2 = tmp2 * 0.5  => 0.5 * ln(...) = atanh(x)
    AscendC::Muls(tmp2, tmp2, (T)0.5, currentNum);

    // 释放输入缓冲区
    inputQueueX.FreeTensor(xLocal);

    // 将结果写入输出队列
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    AscendC::DataCopy(yLocal, tmp2, currentNum * sizeof(T));

    outputQueueY.template EnQue<T>(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAtanh
#endif // ATANH_H
