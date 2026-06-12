/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// Ascend C Kernel 直调 - Atanh 算子（Host + Main 入口）
// ============================================================================
//
// atanh(x) = 0.5 * ln((1 + x) / (1 - x))
//
// 规格:
//   - 输入: 1 个 FP16 tensor (ND 格式, 4 维 shape [N4,N3,N2,N1])
//   - 输出: 1 个 FP16 tensor (与输入同 shape)
//   - 输入范围: |x| < 1 (超出此范围返回 nan)
// ============================================================================

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "../op_kernel/atanh_tiling.h"
#include "data_utils.h"

// 引用拆出的纯 kernel 文件
#include "../op_kernel/atanh_kernel.asc"

using namespace std;

struct ArgInfo {
    string fileName;
    size_t length;
};

// KernelCall 封装 - 处理完整的核函数调用流程
void KernelCall(uint32_t blockNum, aclrtStream stream,
    vector<ArgInfo> &inputsInfo, vector<ArgInfo> &outputsInfo,
    uint8_t *tilingData)
{
    vector<uint8_t *> inputHost(inputsInfo.size());
    vector<uint8_t *> inputDevice(inputsInfo.size());
    vector<uint8_t *> outputHost(outputsInfo.size());
    vector<uint8_t *> outputDevice(outputsInfo.size());
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&tilingDevice, sizeof(AtanhTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDevice, sizeof(AtanhTilingData), tilingData,
        sizeof(AtanhTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    for (size_t i = 0; i < inputsInfo.size(); i++) {
        aclrtMallocHost((void **)&inputHost[i], inputsInfo[i].length);
        aclrtMalloc((void **)&inputDevice[i], inputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputsInfo[i].fileName, inputsInfo[i].length, inputHost[i], inputsInfo[i].length);
        aclrtMemcpy(inputDevice[i], inputsInfo[i].length, inputHost[i],
            inputsInfo[i].length, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    for (size_t i = 0; i < outputsInfo.size(); i++) {
        aclrtMallocHost((void **)&outputHost[i], outputsInfo[i].length);
        aclrtMalloc((void **)&outputDevice[i], outputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 调用 kernel: atanh 是单输入单输出算子
    atanh_kernel<<<blockNum, nullptr, stream>>>(
        inputDevice[0], outputDevice[0], tilingDevice);
    aclrtSynchronizeStream(stream);

    aclrtFree(tilingDevice);
    for (size_t i = 0; i < outputsInfo.size(); i++) {
        aclrtMemcpy(outputHost[i], outputsInfo[i].length, outputDevice[i],
            outputsInfo[i].length, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(outputsInfo[i].fileName, outputHost[i], outputsInfo[i].length);
        aclrtFree(outputDevice[i]);
        aclrtFreeHost(outputHost[i]);
    }

    for (size_t i = 0; i < inputsInfo.size(); i++) {
        aclrtFree(inputDevice[i]);
        aclrtFreeHost(inputHost[i]);
    }
}

int32_t main(int32_t argc, char *argv[])
{
    aclInit(nullptr);
    int32_t deviceId = 0;
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed, ret=%d\n", ret);
        return ret;
    }

    // 获取设备 AICore 核数
    int64_t availableCoreNum = 0;
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        printf("aclrtGetDeviceInfo failed, ret=%d\n", ret);
        aclrtResetDevice(deviceId);
        return ret;
    }

    // 数据规格: 4 维 shape [N4, N3, N2, N1], 例如 [8, 4, 256, 4096]
    // totalLength = N4 * N3 * N2 * N1
    uint32_t totalLength = 8 * 4 * 256 * 4096;  // = 33,554,432 个元素
    size_t inputByteSize = totalLength * sizeof(half);
    size_t outputByteSize = totalLength * sizeof(half);

    // 设置 Tiling 数据
    AtanhTilingData tiling;
    tiling.totalLength = totalLength;
    uint64_t totalTiles = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    uint64_t tilesPerCore = (totalTiles + availableCoreNum - 1) / availableCoreNum;
    uint32_t blockNum = tiling.blockNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    tiling.numPerCore = tilesPerCore * TILE_LENGTH;
    tiling.tailNumLastCore = totalLength - tiling.numPerCore * (blockNum - 1);

    printf("Atanh Kernel Configuration:\n");
    printf("  Shape: [8, 4, 256, 4096] (total=%u elements)\n", totalLength);
    printf("  DataType: float16\n");
    printf("  Format: ND\n");
    printf("  AICores: %ld\n", availableCoreNum);
    printf("  BlockNum: %u\n", blockNum);
    printf("  TILE_LENGTH: %u\n", TILE_LENGTH);
    printf("  ElementsPerCore: %lu\n", tiling.numPerCore);
    printf("  TailElementsLastCore: %lu\n", tiling.tailNumLastCore);

    vector<ArgInfo> inputsInfo = {
        {"./input/input_x.bin", inputByteSize}
    };
    vector<ArgInfo> outputsInfo = {
        {"./output/output.bin", outputByteSize}
    };

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    KernelCall(blockNum, stream, inputsInfo, outputsInfo, (uint8_t *)&tiling);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("Atanh kernel execution completed.\n");
    return 0;
}
