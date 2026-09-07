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
 * Tanh 算子 Tiling 参数。
 * host 侧 TilingFunc 负责填充，kernel 侧通过 GET_TILING_DATA 读取；
 * 字段含义（与 add_example 的 totalNum/blockFactor 思路对应）：
 *   totalLength : 输入 x 的总元素个数（多核切分的总数据量）
 *   tileNum     : 每个核上把本核数据切成的“超块”个数，
 *                 每个超块再按 BUFFER_NUM(=2) 双缓冲切成小块进行流水搬运计算
 */
struct TanhCustomTilingData {
    uint32_t totalLength; // 输入数据总元素个数
    uint32_t tileNum;     // 每核数据切分的超块个数
};
#endif
