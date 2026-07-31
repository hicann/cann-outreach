/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "../mod/arch35/mod_dag.h"
#include "arch35/truncate_mod_struct.h"
#include "arch35/truncate_mod_optimized.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace Ops::Base;

// Tilingkey定义:
// 0: other为标量，使用标量优化路径
// 1: other为张量，使用标准广播路径 + Double Buffer优化

template <uint64_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 根据tilingkey选择优化路径
    if constexpr (schMode == 0) {
        // 标量优化路径: other为标量
        if constexpr (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value) {
            truncate_mod_scalar_optimized<DTYPE_X1>(x1, x2, y, workspace, tiling);
        } else if constexpr (std::is_same<DTYPE_X1, float>::value) {
            truncate_mod_scalar_optimized<float>(x1, x2, y, workspace, tiling);
        } else if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
            truncate_mod_scalar_optimized<int8_t>(x1, x2, y, workspace, tiling);
        } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
            truncate_mod_scalar_optimized<uint8_t>(x1, x2, y, workspace, tiling);
        } else if constexpr (std::is_same<DTYPE_X1, int32_t>::value) {
            truncate_mod_scalar_optimized<int32_t>(x1, x2, y, workspace, tiling);
        } else if constexpr (std::is_same<DTYPE_X1, int64_t>::value) {
            truncate_mod_scalar_optimized<int64_t>(x1, x2, y, workspace, tiling);
        }
    } else {
        // 张量路径: 使用标准广播 + Double Buffer优化
        if constexpr (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value) {
            using OpDag = ModOp::ModFloatWithCastOp<DTYPE_X1>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, float>::value) {
            using OpDag = ModOp::ModFloatOp<float>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
            using OpDag = ModOp::ModIntWithCastOp<int8_t, int16_t>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
            using OpDag = ModOp::ModIntWithCastOp<uint8_t, uint16_t>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, int32_t>::value) {
            using OpDag = ModOp::ModIntOp<int32_t>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, int64_t>::value) {
            using OpDag = ModOp::ModInt64Op<int64_t>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    }
}