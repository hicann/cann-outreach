/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

// 单次搬运/处理的数据量（元素数）
// FP16: 一个元素 2 字节，TILE_LENGTH=8192 对应 16KB UB 空间
// FP32 中间计算: 每个 buffer 8192*4=32KB，两个 buffer 共 64KB
// 加上 I/O queue (FP16)，总 UB 占用约 96KB，UB 192KB 足够
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

// Tiling 数据结构 - 向核函数传递的运行时参数
struct AtanhTilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
};
