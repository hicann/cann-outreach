/**
 * @file main.cpp
 * @brief SubCustomTemplate — 内测用例 (算子包内编译)
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include <algorithm>
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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                    void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return SUCCESS;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs,
                      aclrtStream stream, int32_t deviceId, void *workspaceAddr = nullptr)
{
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 测试数据: shape = {8, 2048} (与 Lesson2 原始设计一致)
    std::vector<int64_t> inputXShape = {8, 2048};
    std::vector<int64_t> inputYShape = {8, 2048};
    std::vector<int64_t> outputZShape = {8, 2048};

    uint64_t totalElements = 8 * 2048;  // 16384
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    // 初始化测试数据: x = 5.0, y = 2.0, golden: z = x - y = 3.0
    std::vector<float> inputXHostData(totalElements, 5.0f);
    std::vector<float> inputYHostData(totalElements, 2.0f);
    std::vector<float> outputZHostData(totalElements, 0.0f);

    std::vector<void *> tensors = {inputX, inputY, outputZ};
    std::vector<void *> deviceAddrs = {inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr};

    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr,
                          aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr,
                          aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr,
                          aclDataType::ACL_FLOAT, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    // ACLNN API 调用
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnSubCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ,
                                                  &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSubCustomTemplateGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }

    ret = aclnnSubCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubCustomTemplate failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 读取结果并验证
    uint64_t size = totalElements;
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                      outputZDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    // Golden: z = x - y = 5.0 - 2.0 = 3.0
    std::vector<float> goldenData(size, 3.0f);

    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("%.1f ", resultData[i]);
    }
    LOG_PRINT("\n");

    if (std::equal(resultData.begin(), resultData.end(), goldenData.begin())) {
        LOG_PRINT("test pass\n");
    } else {
        LOG_PRINT("test failed\n");
        return FAILED;
    }
    return SUCCESS;
}
