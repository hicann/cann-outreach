/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFT_SHRINK_GRAD_TILING_DATA_H
#define SOFT_SHRINK_GRAD_TILING_DATA_H

struct SoftShrinkGradTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
    float lambd = 0.5f;
};

#endif // SOFT_SHRINK_GRAD_TILING_DATA_H
