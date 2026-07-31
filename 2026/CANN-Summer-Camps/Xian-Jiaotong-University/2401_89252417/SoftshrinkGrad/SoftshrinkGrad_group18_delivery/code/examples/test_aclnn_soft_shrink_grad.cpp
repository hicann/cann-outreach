/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_soft_shrink_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}


static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    aclTensor* input_grad = nullptr;
    void* input_gradDeviceAddr = nullptr;
    aclTensor* input_x = nullptr;
    void* input_xDeviceAddr = nullptr;
    aclTensor* output_y = nullptr;
    void* output_yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<int64_t> input_gradShape = {5};
    std::vector<float> input_gradHostData(5, 1);
    std::vector<int64_t> input_xShape = {5};
    std::vector<float> input_xHostData(5, 1);
    std::vector<int64_t> output_yShape = {5};
    std::vector<float> output_yHostData(5, 0);

    // 构造输入 tensor
    ret = CreateAclTensor(input_gradHostData, input_gradShape, &input_gradDeviceAddr, aclDataType::ACL_FLOAT16, &input_grad);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(input_xHostData, input_xShape, &input_xDeviceAddr, aclDataType::ACL_FLOAT16, &input_x);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    // 构造输出 tensor
    ret = CreateAclTensor(output_yHostData, output_yShape, &output_yDeviceAddr, aclDataType::ACL_FLOAT16, &output_y);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    // 调用 aclnnSoftShrinkGradGetWorkspaceSize 第一段接口
    ret = aclnnSoftShrinkGradGetWorkspaceSize(input_grad, input_x, 0.5f, output_y, &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("aclnnSoftShrinkGradGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    // 申请 workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }

    // 调用 aclnnSoftShrinkGrad 第二段接口
    ret = aclnnSoftShrinkGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("aclnnSoftShrinkGrad failed. ERROR: %d\n", ret); goto cleanup);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto cleanup);

cleanup:
    if (input_grad != nullptr) {
        aclDestroyTensor(input_grad);
    }
    if (input_gradDeviceAddr != nullptr) {
        aclrtFree(input_gradDeviceAddr);
    }
    if (input_x != nullptr) {
        aclDestroyTensor(input_x);
    }
    if (input_xDeviceAddr != nullptr) {
        aclrtFree(input_xDeviceAddr);
    }
    if (output_y != nullptr) {
        aclDestroyTensor(output_y);
    }
    if (output_yDeviceAddr != nullptr) {
        aclrtFree(output_yDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (executor != nullptr) {
        aclDestroyOpExecutor(executor);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
