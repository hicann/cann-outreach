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
 * \file relu_example.h
 * \brief ReluExample 算子 kernel 类定义（arch22 架构）
 *
 * 一元逐元素算子，计算 y = max(0, x)。
 * 模板参数：
 *   - T: 数据类型（half）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 执行流程：
 *   CopyIn(GM→UB) → Compute(Relu) → CopyOut(UB→GM)
 */

#ifndef RELU_EXAMPLE_H
#define RELU_EXAMPLE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_example_tiling_data.h"
#include "relu_example_tiling_key.h"

namespace NsReluExample {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class ReluExample {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline ReluExample(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluExampleTilingData* tilingData);
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
__aicore__ inline void ReluExample<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const ReluExampleTilingData* tilingData)
{
    // 当前核处理的元素数（多核均分）
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ReluExample<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
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
__aicore__ inline void ReluExample<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
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
__aicore__ inline void ReluExample<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    // 核心计算：y = Relu(x) = max(0, x)
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ReluExample<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsReluExample
#endif // RELU_EXAMPLE_H
