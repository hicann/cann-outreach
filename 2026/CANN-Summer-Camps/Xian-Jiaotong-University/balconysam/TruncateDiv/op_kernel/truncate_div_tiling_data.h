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
 * \file truncate_div_tiling_data.h
 * \brief tiling data struct for TruncateDiv operator
 * NOTE: This file is shared between op_host/ and op_kernel/.
 *       Host writes it, kernel reads it. Fields must stay in sync.
 */

#ifndef TRUNCATE_DIV_TILING_DATA_H
#define TRUNCATE_DIV_TILING_DATA_H

#include <cstdint>

static constexpr uint32_t TRUNCATE_DIV_MAX_DIM = 8;

struct TruncateDivTilingData {
    // --- Output space ---
    uint64_t totalLength;          // total number of output elements
    uint64_t coreLength;           // elements per core (aligned)
    uint32_t tileLength;           // elements per UB tile
    uint32_t usedCoreNum;          // actual number of AIV cores used

    // --- Mode encoding ---
    uint32_t dtypeMode;            // 0=fp16, 1=fp32, 2=bf16, 3=int8, 4=uint8, 5=int32
    uint32_t broadcastMode;        // 0=same shape, 1=x1 scalar, 2=x2 scalar, 3=unilateral, 4=general

    // --- Standardised output shape / input stride tables (right-aligned) ---
    uint64_t outputShape[TRUNCATE_DIV_MAX_DIM];
    uint64_t x1Stride[TRUNCATE_DIV_MAX_DIM];
    uint64_t x2Stride[TRUNCATE_DIV_MAX_DIM];
};

#endif  // TRUNCATE_DIV_TILING_DATA_H
