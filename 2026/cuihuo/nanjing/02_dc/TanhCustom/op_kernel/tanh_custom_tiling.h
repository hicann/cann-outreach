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

struct TanhCustomTilingData {
    // 数据总元素数（所有核处理的总量）
    uint32_t totalLength;
    // 每个核负责处理的元素数（块长度）
    uint32_t blockLength;
    // 核内每次 tile 的元素数（tile 长度，需 32B 对齐）
    uint32_t tileLength;
    // 核内 tile 分块数量
    uint32_t tileNum;
};
#endif 
