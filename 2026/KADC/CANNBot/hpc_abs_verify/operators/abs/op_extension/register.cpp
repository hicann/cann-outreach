/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// 注册算子签名 - abs 算子（单输入单输出）
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("abs(Tensor x) -> Tensor");
}

// 绑定 NPU 实现
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("abs", TORCH_FN(ascend_kernel::abs_torch));
}

// 绑定 Meta 实现（torch.compile / fx 需要）
// Meta 函数只推导输出 shape/dtype，不执行实际计算。
at::Tensor abs_meta(const at::Tensor& x)
{
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("abs", &abs_meta);
}

} // namespace
