/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "aclrtlaunch_matmul_leakyrelu_custom.h"
#include "kernel_tiling/kernel_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

extern uint8_t *GenerateTiling();

namespace my_matmul_leakyrelu {
at::Tensor run_matmul_leakyrelu_custom(const at::Tensor &a, const at::Tensor &b, const at::Tensor &bias)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    auto c =
        at::empty({a.sizes()[0], b.sizes()[1]}, at::TensorOptions().dtype(at::kFloat).device(a.options().device()));

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t user_workspace_size = 0;
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    size_t workspace_size = user_workspace_size + system_workspace_size;
    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(a.options().device()));

    size_t tilingFileSize = sizeof(TCubeTiling);
    uint8_t *tilingHost;
    uint8_t *tilingDevice;

    aclrtMallocHost((void **)(&tilingHost), tilingFileSize);
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingHost, tilingFileSize, GenerateTiling(), tilingFileSize, ACL_MEMCPY_HOST_TO_HOST);
    aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef CUSTOM_ASCEND310P
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 1;
#endif
    ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
    (blockDim, acl_stream, const_cast<void *>(a.storage().data()), const_cast<void *>(b.storage().data()),
     const_cast<void *>(bias.storage().data()), const_cast<void *>(c.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
    return c;
}
} // namespace my_matmul_leakyrelu

PYBIND11_MODULE(matmul_leakyrelu_custom, m)
{
    m.doc() = "matmul_leakyrelu_custom pybind11 interfaces"; // optional module docstring
    m.def("run_matmul_leakyrelu_custom", &my_matmul_leakyrelu::run_matmul_leakyrelu_custom, "");
}
