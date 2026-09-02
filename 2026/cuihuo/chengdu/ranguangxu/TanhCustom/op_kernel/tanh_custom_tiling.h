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
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H

#include <cstdint>

// Tiling 参数结构体：由 Host 侧（op_host）计算并写入，Device 侧（op_kernel）读取。
// 字段名称与类型必须与 Host/Device 两端严格保持一致，否则会导致 Tiling 数据解析错误。
struct TanhCustomTilingData {
    uint32_t blockLength;  // 单个核（Block）需要处理的元素总个数
    uint32_t tileNum;      // 单个核内划分的 tile 数量
};

#endif  // TANH_CUSTOM_TILING_H
