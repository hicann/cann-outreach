/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file gcd_proto.h
 * @brief Gcd 算子图模式 Proto 定义
 */
#ifndef GCD_PROTO_H_
#define GCD_PROTO_H_

#include "graph/operator.h"
#include "register/op_impl_registry.h"

namespace ge {

REGISTER_OP(GcdOp)
    .Input("self", DT_FLOAT16)
    .Input("other", DT_FLOAT16)
    .Output("out", DT_FLOAT16)
    .OP_END()

} // namespace ge

#endif // GCD_PROTO_H_
