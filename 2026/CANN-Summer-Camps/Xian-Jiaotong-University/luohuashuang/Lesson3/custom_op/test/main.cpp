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
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_add_custom_template.h"

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
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

    // 1. 定义 Shape
    std::vector<int64_t> inputXShape = {45, 20480};
    std::vector<int64_t> inputYShape = {45, 20480};
    std::vector<int64_t> outputZShape = {45, 20480};

    // 2. 准备 Host 数据
    std::vector<float> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<float> inputYHostData(inputYShape[0] * inputYShape[1]);
    std::vector<float> outputZHostData(outputZShape[0] * outputZShape[1]);
    for (int i = 0; i < inputXShape[0] * inputXShape[1]; ++i) {
        inputXHostData[i] = 1.0;
        inputYHostData[i] = 2.0;
        outputZHostData[i] = 0.0;
    }

    // 3. 定义 Device 指针和 Tensor 句柄（初始化为 nullptr）
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    // 4. 创建 inputX
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("Create inputX failed. ERROR: %d\n", ret);
              DestroyResources({}, {}, stream, deviceId); return FAILED);

    // 5. 创建 inputY
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("Create inputY failed. ERROR: %d\n", ret);
              DestroyResources({inputX}, {inputXDeviceAddr}, stream, deviceId); return FAILED);

    // 6. 创建 outputZ
    ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr, aclDataType::ACL_FLOAT, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("Create outputZ failed. ERROR: %d\n", ret);
              DestroyResources({inputX, inputY}, {inputXDeviceAddr, inputYDeviceAddr}, stream, deviceId); return FAILED);

    // ===== 关键修复：在所有 CreateAclTensor 成功后，统一构造资源列表 =====
    std::vector<void *> tensors = {inputX, inputY, outputZ};
    std::vector<void *> deviceAddrs = {inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr};

    // 7. 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnAddCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclnnAddCustomTemplateGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, 
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }

    ret = aclnnAddCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclnnAddCustomTemplate failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 8. 拷贝结果
    auto size = GetShapeSize(outputZShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputZDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("copy result failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 9. 释放资源
    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    // 10. 验证结果
    std::vector<float> goldenData(size, 3.0);
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