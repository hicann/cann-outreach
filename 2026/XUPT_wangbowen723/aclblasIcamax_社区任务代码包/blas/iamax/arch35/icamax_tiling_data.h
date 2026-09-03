/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
struct IcamaxTilingData {
    uint32_t totalN;     // total complex element count
    uint32_t perCoreN;   // complex elements per core (first useCoreNum-1 cores)
    uint32_t lastCoreN;  // complex elements for the last core
    uint32_t useCoreNum; // actual core count used
    uint32_t nthreads;   // SIMT thread count per core
    uint32_t incx;       // stride of x in complex-element units
};
