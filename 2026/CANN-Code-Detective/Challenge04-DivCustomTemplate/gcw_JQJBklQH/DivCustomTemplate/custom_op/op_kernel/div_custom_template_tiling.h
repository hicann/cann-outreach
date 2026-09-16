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


#ifndef DIV_CUSTOM_TEMPLATE_TILING_H
#define DIV_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

// Tiling 参数结构体：由 Host 侧 TilingFunc 填充，通过 tiling 参数传递给核函数
struct DivCustomTemplateTilingData {
    uint32_t size;      // 输入张量的总元素个数
    uint32_t tileNum;   // 每个核内的分块次数（UB 切分块数）
};

#endif // DIV_CUSTOM_TEMPLATE_TILING_H
