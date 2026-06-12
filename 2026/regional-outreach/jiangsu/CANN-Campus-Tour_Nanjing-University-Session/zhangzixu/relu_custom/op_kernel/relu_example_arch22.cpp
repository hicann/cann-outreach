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
 * \file relu_example_arch22.cpp
 * \brief ReluExample 算子 kernel 入口（arch22 架构）
 *
 * 模板参数：
 *   - D_T_X: 数据类型（由 ASCENDC_TPL_DATATYPE_DECL 定义）
 *   - BUFFER_MODE: 缓冲模式（0/1）
 */

#include "arch22/relu_example.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void relu_example(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluExampleTilingData, tilingData, tiling);
    NsReluExample::ReluExample<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
