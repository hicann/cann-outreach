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
 * \file truncate_mod.cpp
 * \brief TruncateMod 算子 kernel 入口
 */

#include "truncate_mod.h"

template <uint32_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X1)
    NsTruncateMod::TruncateMod<DTYPE_X1> op;
#else
    NsTruncateMod::TruncateMod<int32_t> op;
#endif

    op.Init(x1, x2, y, &tilingData);
    op.Process();
}
