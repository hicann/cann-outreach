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
 * \file gcd.h
 * \brief GCD 算子 kernel 类定义（arch22 架构 - ascend910b）
 *
 * 计算逻辑：对于每对 float16 输入 (a, b)：
 *   1. 截断为 int32：ia = trunc(a), ib = trunc(b)
 *   2. 取绝对值
 *   3. 计算最大公约数（辗转相除法）
 *   4. 将结果转回 float16 输出
 *
 * 模板参数：
 *   - T: 数据类型（half/float16）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 */
#ifndef GCD_H
#define GCD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gcd_tiling_data.h"
#include "gcd_tiling_key.h"

namespace NsGcd {

using namespace AscendC;

// 标量 GCD 计算函数（辗转相除法）
__aicore__ inline int32_t GcdScalar(int32_t a, int32_t b)
{
    // 取绝对值
    a = (a < 0) ? -a : a;
    b = (b < 0) ? -b : b;

    // 处理边界情况
    if (a == 0) return b;
    if (b == 0) return a;

    // 辗转相除法
    while (b != 0) {
        int32_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

template <typename T, int BUFFER_MODE>
class Gcd {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Gcd(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out, const GcdTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueOut;

    GlobalTensor<T> inputGMSelf;
    GlobalTensor<T> inputGMOther;
    GlobalTensor<T> outputGMOut;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Init(
    GM_ADDR self, GM_ADDR other, GM_ADDR out, const GcdTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMSelf.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    inputGMOther.SetGlobalBuffer((__gm__ T*)other + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMOut.SetGlobalBuffer((__gm__ T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueOther, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueOut, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> selfLocal = inputQueueSelf.template AllocTensor<T>();
    LocalTensor<T> otherLocal = inputQueueOther.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(selfLocal, inputGMSelf[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(otherLocal, inputGMOther[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inputQueueSelf.EnQue(selfLocal);
    inputQueueOther.EnQue(otherLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueueOut.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGMOut[progress * ubLength_], outLocal, copyParams);

    outputQueueOut.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> selfLocal = inputQueueSelf.template DeQue<T>();
    LocalTensor<T> otherLocal = inputQueueOther.template DeQue<T>();
    LocalTensor<T> outLocal = outputQueueOut.template AllocTensor<T>();

    // 逐元素计算 GCD
    // 对每对 float16 值：截断为 int32 → 计算 GCD → 转回 float16
    for (int64_t i = 0; i < currentNum; i++) {
        // 将 float16 截断为 int32
        int32_t a = static_cast<int32_t>(selfLocal.GetValue(i));
        int32_t b = static_cast<int32_t>(otherLocal.GetValue(i));

        // 计算 GCD
        int32_t r = GcdScalar(a, b);

        // 结果转回 float16
        outLocal.SetValue(i, static_cast<T>(r));
    }

    outputQueueOut.template EnQue<T>(outLocal);
    inputQueueSelf.FreeTensor(selfLocal);
    inputQueueOther.FreeTensor(otherLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsGcd
#endif // GCD_H
