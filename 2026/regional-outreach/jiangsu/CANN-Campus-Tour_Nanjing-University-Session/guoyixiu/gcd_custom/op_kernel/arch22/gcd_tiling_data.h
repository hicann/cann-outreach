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
 * \file gcd_tiling_data.h
 * \brief Gcd 算子 Tiling 数据结构（二元 broadcast 算子）
 *
 * 数据成员说明：
 *   - totalNum:    广播后输出 tensor 的总元素数
 *   - blockFactor: 每个核处理的元素数
 *   - ubFactor:    每次 UB 循环处理的元素数
 *   - dimNum:      shape 维度数（最多 8）
 *   - selfShape/otherShape/outShape: 各 tensor 的 shape（尾部对齐广播规则）
 *   - selfStrides/otherStrides/outStrides: 各 tensor 的 stride（用于 flat index 计算）
 *   - selfNum/otherNum: 各输入的实际元素数（用于空 tensor 判断）
 */

#ifndef GCD_TILING_DATA_H
#define GCD_TILING_DATA_H

#include <cstdint>

constexpr int64_t GCD_MAX_SHAPE_DIMS = 8;

struct GcdTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;

    int64_t dimNum;
    int64_t selfShape[GCD_MAX_SHAPE_DIMS];
    int64_t otherShape[GCD_MAX_SHAPE_DIMS];
    int64_t outShape[GCD_MAX_SHAPE_DIMS];
    int64_t selfStrides[GCD_MAX_SHAPE_DIMS];
    int64_t otherStrides[GCD_MAX_SHAPE_DIMS];
    int64_t outStrides[GCD_MAX_SHAPE_DIMS];
    int64_t selfNum;
    int64_t otherNum;
};

#endif // GCD_TILING_DATA_H
