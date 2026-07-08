/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sub_custom_template.h"

#define SUCCESS 0
#define FAILED 1

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Fixed code, acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
aclDataType GetAclDataType();

template <>
aclDataType GetAclDataType<aclFloat16>()
{
    return ACL_FLOAT16;
}

template <>
aclDataType GetAclDataType<float>()
{
    return ACL_FLOAT;
}

template <typename T>
const char *GetTypeName();

template <>
const char *GetTypeName<aclFloat16>()
{
    return "float16";
}

template <>
const char *GetTypeName<float>()
{
    return "float32";
}

template <typename T>
T CastValue(float value);

template <>
aclFloat16 CastValue<aclFloat16>(float value)
{
    return aclFloatToFloat16(value);
}

template <>
float CastValue<float>(float value)
{
    return value;
}

template <typename T>
float ToFloat(T value);

template <>
float ToFloat<aclFloat16>(aclFloat16 value)
{
    return aclFloat16ToFloat(value);
}

template <>
float ToFloat<float>(float value)
{
    return value;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // Call aclCreateTensor to create a aclTensor object
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

void DestroyResources(const std::vector<aclTensor *> &tensors, const std::vector<void *> &deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    // Release aclTensor and device
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(tensors[i]);
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    // Destroy stream and reset device
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int RunCase(int32_t deviceId)
{
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    std::vector<int64_t> inputXShape = {8, 2048};
    std::vector<int64_t> inputYShape = {8, 2048};
    std::vector<int64_t> outputZShape = {8, 2048};
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    std::vector<T> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<T> inputYHostData(inputYShape[0] * inputYShape[1]);
    std::vector<T> outputZHostData(outputZShape[0] * outputZShape[1]);
    for (int i = 0; i < inputXShape[0] * inputXShape[1]; ++i) {
        inputXHostData[i] = CastValue<T>(1.0f);
        inputYHostData[i] = CastValue<T>(2.0f);
        outputZHostData[i] = CastValue<T>(0.0f);
    }
    std::vector<aclTensor *> tensors = {nullptr, nullptr, nullptr};
    std::vector<void *> deviceAddrs = {nullptr, nullptr, nullptr};
    // Create inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, GetAclDataType<T>(), &inputX);
    tensors[0] = inputX;
    deviceAddrs[0] = inputXDeviceAddr;
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    // Create inputY aclTensor
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, GetAclDataType<T>(), &inputY);
    tensors[1] = inputY;
    deviceAddrs[1] = inputYDeviceAddr;
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    // Create outputZ aclTensor
    ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr, GetAclDataType<T>(), &outputZ);
    tensors[2] = outputZ;
    deviceAddrs[2] = outputZDeviceAddr;
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSubCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubCustomTemplateGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }
    // Execute the custom operator
    ret = aclnnSubCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubCustomTemplate failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    auto size = GetShapeSize(outputZShape);
    std::vector<T> resultData(size, CastValue<T>(0.0f));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputZDeviceAddr,
                      size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    std::vector<T> goldenData(size, CastValue<T>(-1.0f));

    LOG_PRINT("[%s] result is:\n", GetTypeName<T>());
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("%.1f ", ToFloat(resultData[i]));
    }
    LOG_PRINT("\n");
    const bool pass = std::equal(resultData.begin(), resultData.end(), goldenData.begin(), [](const T &lhs, const T &rhs) {
        return std::fabs(ToFloat(lhs) - ToFloat(rhs)) < 1e-3f;
    });

    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    if (!pass) {
        LOG_PRINT("test failed\n");
        return FAILED;
    }
    LOG_PRINT("test pass\n");
    return SUCCESS;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    const int32_t deviceId = 0;

    int ret = RunCase<aclFloat16>(deviceId);
    CHECK_RET(ret == SUCCESS, return FAILED);

    ret = RunCase<float>(deviceId);
    CHECK_RET(ret == SUCCESS, return FAILED);

    return SUCCESS;
}
