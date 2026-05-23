/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

// ⚠️ Stream 同步反模式（详见 torch-ascendc-op-extension SKILL.md「Stream 同步模式」和 references/anti_patterns.md）：
//   ❌ stream(false) + 函数调用 → 乱序：不清 queue，kernel 先于之前操作执行
//   ❌ lambda 内传 NPUStream + OpCommand → 死锁：queue 等 lambda，lambda 等 queue 空
//   ❌ zeros_like 创建输出 → 乱序：zeros_like 入 queue 但 kernel 不入 queue，用 empty_like
//   ✅ stream(true) + 函数调用 → 清 queue，安全（本文件使用此模式）

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/abs_tiling.h"

// extern 声明 kernel 入口 - abs 算子（单输入单输出）
extern "C" void abs_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                           uint8_t *x, uint8_t *y, uint8_t *tiling);

// 算子 PyTorch 实现
namespace ascend_kernel {

at::Tensor abs_torch(const at::Tensor& x)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only FP32 supported");
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");

    at::Tensor y = at::empty_like(x);

    int64_t totalElements = x.numel();
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");

    // stream(true) 在返回 ACL stream 前会清 queue，确保与之前 NPU 操作的正确同步
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // Tiling 计算（按 DESIGN.md §2.1 和 §2.2）
    AbsTilingData tiling;
    tiling.totalLength = static_cast<uint32_t>(totalElements);
    tiling.dtypeSize = sizeof(float);
    
    uint32_t dim0 = static_cast<uint32_t>(totalElements);
    
    // Step 1: 计算核数
    uint32_t minDtypeBits = sizeof(float) * 8;
    uint32_t coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / 
                       MIN_TILING_BITS_SIZE_PER_CORE;
    coreNum = std::min(coreNum, static_cast<uint32_t>(availableCoreNum));
    
    // Step 2: 每个核的基础元素数
    uint32_t blockFormer;
    uint32_t blockNum;
    
    if (dim0 < ELEM_ALIGN_FACTOR) {
        // 小 shape 场景
        coreNum = 1;
        blockFormer = dim0;
        blockNum = 1;
    } else {
        blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) / 
                       ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
        blockNum = (dim0 + blockFormer - 1) / blockFormer;
    }
    
    // Step 3: 计算尾部数据
    uint32_t blockTail = dim0 - (blockNum - 1) * blockFormer;
    
    // Step 4: UB 切分
    uint32_t ubSize = 192 * 1024;
    uint32_t bufferDivisor = 4 * sizeof(float);
    uint32_t maxElemNum = ubSize / bufferDivisor;
    uint32_t alignFactor = REPEAT_BYTES / sizeof(float);
    uint32_t ubFormer = (maxElemNum / alignFactor) * alignFactor;
    
    // Step 5: UB 循环次数和尾部
    uint32_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    uint32_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    
    uint32_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    uint32_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
    
    // 设置 Tiling 结构体
    tiling.blockNum = blockNum;
    tiling.blockFormer = blockFormer;
    tiling.blockTail = blockTail;
    tiling.ubFormer = ubFormer;
    tiling.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling.ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling.ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling.ubTailOfTailBlock = ubTailOfTailBlock;

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(AbsTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(AbsTilingData),
        &tiling, sizeof(AbsTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用 kernel（单输入单输出）
    abs_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return y;
}

} // namespace ascend_kernel
