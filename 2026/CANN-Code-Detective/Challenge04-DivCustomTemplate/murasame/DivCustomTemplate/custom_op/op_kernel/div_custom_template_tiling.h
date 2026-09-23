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

// host 侧与 kernel 侧共享的 tiling 数据结构
struct DivCustomTemplateTilingData {
    uint32_t size;  // 参与计算的元素总数（x/y/z 的 shape 展平后的长度）
    uint32_t dtype; // 输入数据类型：0 = float16，1 = float32
};

#endif // DIV_CUSTOM_TEMPLATE_TILING_H
