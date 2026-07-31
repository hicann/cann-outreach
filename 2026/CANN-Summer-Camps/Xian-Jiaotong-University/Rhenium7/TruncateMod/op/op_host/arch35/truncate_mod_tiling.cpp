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
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod算子Tiling实现，支持Double Buffer和标量优化
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "math/mod/op_host/arch35/mod_tiling_arch35.h"

namespace optiling {
constexpr static uint64_t MOD_COMMON_TILING_PRIORITY = 0;

// Tilingkey定义:
// 0: other为标量，使用标量优化路径
// 1: other为张量，使用标准广播路径 + Double Buffer优化

static ge::graphStatus TilingForTruncateMod(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    
    // 获取输入shape信息
    const gert::Tensor* x1Shape = context->GetInputShape(0);
    const gert::Tensor* x2Shape = context->GetInputShape(1);
    
    if (x1Shape == nullptr || x2Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    // 检查other是否为标量 (合轴后dim=1)
    auto x2ShapeData = x2Shape->GetStorageShape();
    bool isScalar = true;
    for (int i = 0; i < x2ShapeData.GetDimNum(); i++) {
        if (x2ShapeData.GetDim(i) != 1) {
            isScalar = false;
            break;
        }
    }
    
    // 设置tilingkey
    // 0: other为标量，使用标量优化路径
    // 1: other为张量，使用标准广播路径
    uint64_t tilingKey = isScalar ? 0 : 1;
    context->SetBlockDim(1); // 默认值，后续会被DoTilingImpl覆盖
    
    // 使用ModTiling进行详细计算
    ge::graphStatus ret = Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 覆盖tilingkey以支持优化路径
    context->SetTilingKey(tilingKey);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
REGISTER_OPS_TILING_TEMPLATE(TruncateMod, ModTiling, MOD_COMMON_TILING_PRIORITY);
IMPL_OP_OPTILING(TruncateMod)
    .Tiling(TilingForTruncateMod)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForTruncateMod);
} // namespace optiling