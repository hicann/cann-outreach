// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// relu_custom PyTorch 集成
// TORCH_LIBRARY 注册 + Tiling 计算 + Kernel launch

#include <torch/extension.h>
#include <torch/torch.h>
#include "kernel/relu_custom_tiling.h"
#include "acl/acl.h"

int64_t GetVectorCoreNum()
{
    int32_t deviceId = 0;
    aclrtGetDevice(&deviceId);
    int64_t vectorCoreNum = 8;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreNum);
    return vectorCoreNum;
}

int32_t GetUbAvailableSize()
{
    return 180 * 1024;
}

torch::Tensor relu_custom_forward(torch::Tensor x)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "Input must be float16");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous (ND format)");

    int32_t numElements = x.numel();
    auto y = torch::empty_like(x, torch::kFloat16);

    aclrtStream stream = c10::cuda::getCurrentCUDAStream().stream();
    if (stream == nullptr) {
        aclrtCreateStream(&stream);
    }

    half *dX = reinterpret_cast<half *>(x.data_ptr<at::Half>());
    half *dY = reinterpret_cast<half *>(y.data_ptr<at::Half>());

    // Tiling 计算
    ReluCustomTilingData tiling;
    int64_t coreNum = GetVectorCoreNum();
    int32_t ubSize = GetUbAvailableSize();
    ComputeReluCustomTiling(tiling, numElements, coreNum, ubSize);

    // Tiling Buffer (Device)
    size_t tilingSize = sizeof(ReluCustomTilingData);
    half *dTiling = nullptr;
    aclrtMalloc((void **)&dTiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dTiling, tilingSize, &tiling, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Kernel Launch
    relu_custom_kernel<<<dX, dY, dTiling, stream>>>(tiling.usedCoreNum);
    aclrtSynchronizeStream(stream);

    aclrtFree(dTiling);

    return y;
}

TORCH_LIBRARY(npu, m) {
    m.def("relu_custom(Tensor x) -> Tensor");
    m.impl("relu_custom", &relu_custom_forward);
}
