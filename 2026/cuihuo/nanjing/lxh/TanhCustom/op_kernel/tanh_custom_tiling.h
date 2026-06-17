/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H
#include <cstdint>

/*
 * TanhCustomTilingData —— Host 与 Kernel 共享的 Tiling 参数结构体
 *
 * 数据流向：
 *   Host 侧 TilingFunc 填充 → 嵌入 Kernel 二进制 → Kernel 侧通过 GET_TILING_DATA 读取
 *
 * 字段说明：
 *   totalLength: 所有 AI Core 需要处理的总元素个数（跨所有 Core）
 *                在 Kernel 中会除以 GetBlockNum() 得到每个 Core 的处理量
 *   tileNum:     每个 AI Core 内部的逻辑 Tile 数量
 *                配合 BUFFER_NUM=2 做双缓冲，每个 Tick 搬 tileLength 个元素
 */
struct TanhCustomTilingData {
    uint32_t totalLength;  // 总元素个数（所有 Core 合计），Init 中用于计算 per-core 长度
    uint32_t tileNum;      // 每个 Core 内部的 Tile 数量，控制 pipeline 循环次数
};
#endif
