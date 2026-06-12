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
 * \file gcd_def.cpp
 * \brief GCD 算子定义，声明输入/输出和算子配置
 *
 * 算子规格：
 *   - 输入：self(x1), other(x2) — 2 个必选输入，float16，ND
 *   - 输出：out(y) — 1 个必选输出，float16，ND
 *   - 语义：out = gcd(trunc(self), trunc(other))
 *   - Broadcast：self 与 other 支持广播，输出 shape 为 broadcast 结果
 */
#include "register/op_def_registry.h"

namespace ops {

class Gcd : public OpDef {
public:
    explicit Gcd(const char* name) : OpDef(name)
    {
        this->Input("self")                                         // 输入 self
            .ParamType(REQUIRED)                                    // 必选输入
            .DataType({ge::DT_FLOAT16})                             // 支持 float16
            .Format({ge::FORMAT_ND})                                // ND 格式
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("other")                                        // 输入 other
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("out")                                         // 输出 out
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // ascend910b 配置
        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);

        // ascend950 配置
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};

OP_ADD(Gcd);

} // namespace ops
