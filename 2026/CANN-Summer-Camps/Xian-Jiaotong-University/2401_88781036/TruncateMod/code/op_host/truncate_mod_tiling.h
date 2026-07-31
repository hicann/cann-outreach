/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNCATE_MOD_HOST_TILING_H
#define TRUNCATE_MOD_HOST_TILING_H

#include <cstdint>

namespace optiling {
struct TruncateModCompileInfo {
    int32_t totalCoreNum = 40;
    int64_t ubSize = 196608;
    bool isRegbase = false;
};
} // namespace optiling

#endif // TRUNCATE_MOD_HOST_TILING_H
