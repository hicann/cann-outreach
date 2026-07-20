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
#include <cstdio>
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
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
T FromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
aclFloat16 FromFloat<aclFloat16>(float value)
{
    return aclFloatToFloat16(value);
}

template <typename T>
float ToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
float ToFloat<aclFloat16>(aclFloat16 value)
{
    return aclFloat16ToFloat(value);
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

void DestroyCaseResources(const std::vector<aclTensor *> &tensors, const std::vector<void *> &deviceAddrs,
                          void *workspaceAddr)
{
    for (auto tensor : tensors) {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
    for (auto addr : deviceAddrs) {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
}

template <typename T>
int RunSubCase(const char *caseName, aclDataType dataType, float eps, aclrtStream stream)
{
    const std::vector<int64_t> shape = {8, 2048};
    const int64_t size = GetShapeSize(shape);

    std::vector<T> inputXHostData(size);
    std::vector<T> inputYHostData(size);
    std::vector<T> outputZHostData(size, FromFloat<T>(0.0f));
    std::vector<float> goldenData(size);

    for (int64_t i = 0; i < size; ++i) {
        float xValue = static_cast<float>((i % 13) - 6) * 0.5f;
        float yValue = static_cast<float>((i % 7) - 3) * 0.25f;
        inputXHostData[i] = FromFloat<T>(xValue);
        inputYHostData[i] = FromFloat<T>(yValue);
        goldenData[i] = xValue - yValue;
    }

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    auto cleanup = [&]() {
        DestroyCaseResources({inputX, inputY, outputZ}, {inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr},
                             workspaceAddr);
    };

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, dataType, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, dataType, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, dataType, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSubCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s GetWorkspaceSize failed. ERROR: %d\n", caseName, ret);
              cleanup(); return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", caseName, ret);
                  cleanup(); return FAILED);
    }

    ret = aclnnSubCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnSubCustomTemplate failed. ERROR: %d\n", caseName, ret);
              cleanup(); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", caseName, ret);
              cleanup(); return FAILED);

    std::vector<T> resultData(size);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outputZDeviceAddr,
                      size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s copy result failed. ERROR: %d\n", caseName, ret);
              cleanup(); return FAILED);

    float maxDiff = 0.0f;
    int64_t firstBadIndex = -1;
    for (int64_t i = 0; i < size; ++i) {
        float diff = std::fabs(ToFloat(resultData[i]) - goldenData[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        if (diff > eps && firstBadIndex < 0) {
            firstBadIndex = i;
        }
    }

    LOG_PRINT("%s result first 10:\n", caseName);
    for (int64_t i = 0; i < 10; ++i) {
        LOG_PRINT("%.4f ", ToFloat(resultData[i]));
    }
    LOG_PRINT("\n");

    cleanup();
    if (firstBadIndex < 0) {
        LOG_PRINT("%s test pass. maxDiff=%.8f\n", caseName, maxDiff);
        return SUCCESS;
    }

    LOG_PRINT("%s test failed at index %ld. result=%.8f golden=%.8f maxDiff=%.8f\n", caseName, firstBadIndex,
              ToFloat(resultData[firstBadIndex]), goldenData[firstBadIndex], maxDiff);
    return FAILED;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    ret = RunSubCase<aclFloat16>("float16", aclDataType::ACL_FLOAT16, 1e-3f, stream);
    int finalRet = ret;
    if (ret == SUCCESS) {
        ret = RunSubCase<float>("float32", aclDataType::ACL_FLOAT, 1e-6f, stream);
        finalRet = ret;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return finalRet == SUCCESS ? SUCCESS : FAILED;
}
