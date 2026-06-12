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
 * \file gcd_arch35.cpp
 * \brief Gcd 算子 kernel 入口（arch35 架构，二元 broadcast，float16）
 */

#include "arch35/gcd.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void gcd_kernel(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GcdTilingData);
    GET_TILING_DATA_WITH_STRUCT(GcdTilingData, tilingData, tiling);
    NsGcd::Gcd<D_T_X, BUFFER_MODE> op;
    op.Init(self, other, out, &tilingData);
    op.Process();
}
