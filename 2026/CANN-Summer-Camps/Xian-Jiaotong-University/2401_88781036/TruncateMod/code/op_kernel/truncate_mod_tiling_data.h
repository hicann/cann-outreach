/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncate_mod_tiling_data.h
 * \brief tiling data struct
 */

#ifndef TRUNCATE_MOD_TILING_DATA_H
#define TRUNCATE_MOD_TILING_DATA_H

#include <cstdint>

constexpr uint32_t TRUNCATE_MOD_MAX_DIMS = 8U;

struct TruncateModTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t totalDataNum;
    uint32_t tmpTileDataNum;
    uint32_t tmpSmallTailDataNum;
    uint32_t tmpBigTailDataNum;
    uint32_t x1DataNum;
    uint32_t x2DataNum;
    uint32_t outShape[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x1Shape[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x2Shape[TRUNCATE_MOD_MAX_DIMS];
};

#endif // TRUNCATE_MOD_TILING_DATA_H
