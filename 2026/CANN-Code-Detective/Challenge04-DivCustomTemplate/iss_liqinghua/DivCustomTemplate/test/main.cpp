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
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_div_custom_template.h"

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
    for (int64_t dimension : shape) {
        shapeSize *= dimension;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return FAILED;
    }

    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return FAILED;
    }
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const size_t byteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

void DestroyCaseResources(aclTensor *inputX, aclTensor *inputY, aclTensor *outputZ, void *inputXDeviceAddr,
                          void *inputYDeviceAddr, void *outputZDeviceAddr, void *workspaceAddr)
{
    if (inputX != nullptr) {
        aclDestroyTensor(inputX);
    }
    if (inputY != nullptr) {
        aclDestroyTensor(inputY);
    }
    if (outputZ != nullptr) {
        aclDestroyTensor(outputZ);
    }
    if (inputXDeviceAddr != nullptr) {
        aclrtFree(inputXDeviceAddr);
    }
    if (inputYDeviceAddr != nullptr) {
        aclrtFree(inputYDeviceAddr);
    }
    if (outputZDeviceAddr != nullptr) {
        aclrtFree(outputZDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
}

template <typename T, typename FloatToT, typename TToFloat>
int RunDivCase(aclrtStream stream, aclDataType dataType, const char *caseName, FloatToT toHostValue,
               TToFloat toFloatValue)
{
    const std::vector<int64_t> shape = {8, 2048};
    const int64_t elementCount = GetShapeSize(shape);
    std::vector<T> inputXHostData(elementCount, toHostValue(1.0F));
    std::vector<T> inputYHostData(elementCount, toHostValue(2.0F));
    std::vector<T> outputZHostData(elementCount, toHostValue(0.0F));

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    auto cleanup = [&]() {
        DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                             workspaceAddr);
    };

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, dataType, &inputX);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, dataType, &inputY);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, dataType, &outputZ);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnDivCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDivCustomTemplateGetWorkspaceSize failed. ERROR: %d\n", ret); cleanup();
              return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); cleanup();
                  return FAILED);
    }

    ret = aclnnDivCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivCustomTemplate failed. ERROR: %d\n", ret); cleanup();
              return FAILED);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); cleanup();
              return FAILED);

    std::vector<T> resultData(elementCount);
    const size_t resultBytes = resultData.size() * sizeof(T);
    ret = aclrtMemcpy(resultData.data(), resultBytes, outputZDeviceAddr, resultBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); cleanup();
              return FAILED);

    cleanup();
    std::vector<T> goldenData(elementCount, toHostValue(0.5F));

    LOG_PRINT("%s result is:\n", caseName);
    for (int64_t index = 0; index < 10; ++index) {
        LOG_PRINT("%.1f ", toFloatValue(resultData[index]));
    }
    LOG_PRINT("\n");
    if (!std::equal(resultData.begin(), resultData.end(), goldenData.begin())) {
        LOG_PRINT("%s test failed\n", caseName);
        return FAILED;
    }
    LOG_PRINT("%s test pass\n", caseName);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    ret = RunDivCase<aclFloat16>(stream, aclDataType::ACL_FLOAT16, "float16",
                                 [](float value) { return aclFloatToFloat16(value); },
                                 [](aclFloat16 value) { return aclFloat16ToFloat(value); });
    if (ret == SUCCESS) {
        ret = RunDivCase<float>(stream, aclDataType::ACL_FLOAT, "float32", [](float value) { return value; },
                                [](float value) { return value; });
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
