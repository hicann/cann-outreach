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
 * @file gcd.h
 * @brief ACLNN L0 API 接口声明 - 二元最大公约数算子（float16，支持 broadcast）
 */

#ifndef OP_API_INC_LEVEL0_GCD_H_
#define OP_API_INC_LEVEL0_GCD_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Gcd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_GCD_H_
