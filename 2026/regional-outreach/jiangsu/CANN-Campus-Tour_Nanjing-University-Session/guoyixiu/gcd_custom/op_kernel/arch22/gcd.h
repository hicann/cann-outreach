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
 * \brief Gcd 算子 kernel 类定义（arch22，二元 broadcast，float16）
 *
 * 核心算法：欧几里得辗转相减法（Euclidean algorithm by subtraction）
 * 适用于 float16，支持 broadcast shape 映射。
 *
 * 广播索引映射规则：
 *   对于输出位置 outIdx[d]，对应输入位置：
 *     selfIdx[d] = (selfShape[d] == 1) ? 0 : outIdx[d]
 *     otherIdx[d] = (otherShape[d] == 1) ? 0 : outIdx[d]
 */

#ifndef GCD_H
#define GCD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gcd_tiling_data.h"
#include "gcd_tiling_key.h"

namespace NsGcd {

using namespace AscendC;
using half = AscendC::LocalTensor<float16>::ElementType;

/*!
 * \brief 计算两个 float16 值的最大公约数（辗转相减法）
 *
 * 算法：
 *   gcd(a, b):
 *     a = abs(a), b = abs(b)
 *     if a == 0: return b
 *     if b == 0: return a
 *     while a != b:
 *       if a > b: a = a - b
 *       else: b = b - a
 *     return a
 *
 * 注意：float16 的精度有限，对于非常大的数值或精度要求高的场景，
 * 循环次数受值的大小影响。建议输入值范围在 [-2048, 2048] 以内。
 */
static __aicore__ inline half GcdScalar(half a, half b)
{
    // 取绝对值
    a = (a < (half)0.0f) ? -a : a;
    b = (b < (half)0.0f) ? -b : b;

    // 处理零值
    if (a == (half)0.0f) return b;
    if (b == (half)0.0f) return a;

    // 辗转相减法（欧几里得算法变体）
    // 对 float16 使用减法而非取模，避免除零和精度问题
    while (a != b) {
        if (a > b) {
            a = a - b;
        } else {
            b = b - a;
        }
    }
    return a;
}

/*!
 * \brief 将 flat index 按 strides 分解为多维索引
 */
static __aicore__ inline void DecomposeIndex(int64_t flat, const int64_t* strides,
                                              int64_t* indices, int64_t dimNum)
{
    int64_t rem = flat;
    for (int64_t d = 0; d < dimNum; d++) {
        if (strides[d] > 0) {
            indices[d] = rem / strides[d];
            rem = rem % strides[d];
        } else {
            indices[d] = 0;
        }
    }
}

/*!
 * \brief 将多维索引按 strides 压平为 flat index
 */
static __aicore__ inline int64_t FlattenIndex(const int64_t* indices,
                                               const int64_t* strides, int64_t dimNum)
{
    int64_t flat = 0;
    for (int64_t d = 0; d < dimNum; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}

template <typename T, int BUFFER_MODE>
class Gcd {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Gcd(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                 const GcdTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 计算一个 UB 块内的 GCD（含广播索引映射）
    __aicore__ inline void ComputeBlock(int64_t blockStart, int64_t currentNum);

private:
    // 流水线队列
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueOut;

    // Global tensor 句柄（指向 GM 中本核的分片起始位置）
    GlobalTensor<T> selfGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;

    // Tiling 数据（包含广播 shape 信息）
    const GcdTilingData* tiling_;

    // 本核处理的数据范围
    int64_t blockLength_;
    int64_t ubLength_;

    // 广播 shape 信息（本地拷贝避免重复加载）
    int64_t dimNum_;
    int64_t selfShape_[GCD_MAX_SHAPE_DIMS];
    int64_t otherShape_[GCD_MAX_SHAPE_DIMS];
    int64_t outShape_[GCD_MAX_SHAPE_DIMS];
    int64_t selfStrides_[GCD_MAX_SHAPE_DIMS];
    int64_t otherStrides_[GCD_MAX_SHAPE_DIMS];
    int64_t outStrides_[GCD_MAX_SHAPE_DIMS];
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                                  const GcdTilingData* tilingData)
{
    tiling_ = tilingData;

    // 计算本核处理范围
    int64_t remainder = tiling_->totalNum - tiling_->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainder > tiling_->blockFactor) ? tiling_->blockFactor : remainder;
    ubLength_ = tiling_->ubFactor;

    // Global buffer 按核分片
    selfGM.SetGlobalBuffer((__gm__ T*)self + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    otherGM.SetGlobalBuffer((__gm__ T*)other + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outGM.SetGlobalBuffer((__gm__ T*)out + tiling_->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 初始化 UB 队列（用作临时缓冲区，保存按广播映射加载的输入数据）
    pipe.InitBuffer(inputQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueOther, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueOut, BUFFER_NUM, ubLength_ * sizeof(T));

    // 本地拷贝广播 shape 信息（避免 kernel 循环中反复读取 tiling 数据）
    dimNum_ = tiling_->dimNum;
    for (int64_t d = 0; d < dimNum_; d++) {
        selfShape_[d] = tiling_->selfShape[d];
        otherShape_[d] = tiling_->otherShape[d];
        outShape_[d] = tiling_->outShape[d];
        selfStrides_[d] = tiling_->selfStrides[d];
        otherStrides_[d] = tiling_->otherStrides[d];
        outStrides_[d] = tiling_->outStrides[d];
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::ComputeBlock(int64_t blockStart, int64_t currentNum)
{
    // 分配 UB 缓冲区
    AscendC::LocalTensor<T> selfLocal = inputQueueSelf.template AllocTensor<T>();
    AscendC::LocalTensor<T> otherLocal = inputQueueOther.template AllocTensor<T>();
    AscendC::LocalTensor<T> outLocal = outputQueueOut.template AllocTensor<T>();

    // 逐元素加载输入并计算 GCD（含广播索引映射）
    for (int64_t i = 0; i < currentNum; i++) {
        int64_t outFlat = blockStart + i;

        // 将输出 flat index 分解为多维索引
        int64_t outIdx[GCD_MAX_SHAPE_DIMS];
        DecomposeIndex(outFlat, outStrides_, outIdx, dimNum_);

        // 映射到 self 的多维索引（广播规则：dim==1 的维度索引为 0）
        int64_t selfIdx[GCD_MAX_SHAPE_DIMS];
        for (int64_t d = 0; d < dimNum_; d++) {
            selfIdx[d] = (selfShape_[d] == 1) ? 0 : outIdx[d];
        }
        int64_t selfFlat = FlattenIndex(selfIdx, selfStrides_, dimNum_);

        // 映射到 other 的多维索引
        int64_t otherIdx[GCD_MAX_SHAPE_DIMS];
        for (int64_t d = 0; d < dimNum_; d++) {
            otherIdx[d] = (otherShape_[d] == 1) ? 0 : outIdx[d];
        }
        int64_t otherFlat = FlattenIndex(otherIdx, otherStrides_, dimNum_);

        // 直接读取 GM 中的输入数据（广播场景不适合连续 DMA 拷贝）
        half a = (half)selfGM(selfFlat);
        half b = (half)otherGM(otherFlat);

        // 计算 GCD
        half result = GcdScalar(a, b);

        // 写入 UB 缓冲区
        outLocal.SetValue(i, (T)result);
    }

    // 将计算结果从 UB 拷贝回 GM
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM[blockStart], outLocal, copyParams);

    // 释放 UB 缓冲区
    inputQueueSelf.FreeTensor(selfLocal);
    inputQueueOther.FreeTensor(otherLocal);
    outputQueueOut.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Gcd<T, BUFFER_MODE>::Process()
{
    // 按 UB 大小分块处理
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        int64_t blockStart = ubLength_ * i;
        ComputeBlock(blockStart, currentNum);
    }
}

} // namespace NsGcd
#endif // GCD_H
