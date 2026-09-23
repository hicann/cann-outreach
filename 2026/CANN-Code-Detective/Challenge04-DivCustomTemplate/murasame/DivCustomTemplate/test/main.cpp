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
    return SUCCESS;
}

void DestroyTensors(std::vector<void *> &tensors, std::vector<void *> &deviceAddrs, void *workspaceAddr = nullptr)
{
    // Release aclTensor and device memory
    // tensors/deviceAddrs 在每次资源创建成功后 push 记录，此处释放的是真实已创建的资源
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
}

// 运行一次 Div 精度验证。T 为 host 侧数据类型（aclFloat16 / float），
// toHostValue 将 float 转换为 T，toFloatValue 将 T 转回 float 用于打印，从而同时覆盖 fp16 与 fp32
template <typename T, typename FloatToT, typename TToFloat>
int RunDivCase(aclrtStream stream, aclDataType dataType, const std::vector<int64_t> &shape, FloatToT toHostValue,
               TToFloat toFloatValue)
{
    auto size = GetShapeSize(shape);
    std::vector<T> inputXHostData(size);
    std::vector<T> inputYHostData(size);
    std::vector<T> outputZHostData(size);
    for (int64_t i = 0; i < size; ++i) {
        inputXHostData[i] = toHostValue(1.0);
        inputYHostData[i] = toHostValue(2.0);
        outputZHostData[i] = toHostValue(0.0);
    }

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    std::vector<void *> tensors;
    std::vector<void *> deviceAddrs;

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, dataType, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensors(tensors, deviceAddrs); return FAILED);
    tensors.push_back(inputX);
    deviceAddrs.push_back(inputXDeviceAddr);

    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, dataType, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensors(tensors, deviceAddrs); return FAILED);
    tensors.push_back(inputY);
    deviceAddrs.push_back(inputYDeviceAddr);

    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, dataType, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensors(tensors, deviceAddrs); return FAILED);
    tensors.push_back(outputZ);
    deviceAddrs.push_back(outputZDeviceAddr);

    // 3. Call the API of the custom operator library
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Calculate the workspace size and allocate memory for it
    ret = aclnnDivCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivCustomTemplateGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyTensors(tensors, deviceAddrs); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyTensors(tensors, deviceAddrs, workspaceAddr); return FAILED);
    }
    // Execute the custom operator
    ret = aclnnDivCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivCustomTemplate failed. ERROR: %d\n", ret);
              DestroyTensors(tensors, deviceAddrs, workspaceAddr); return FAILED);

    // 4. (Fixed code) Synchronize and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyTensors(tensors, deviceAddrs, workspaceAddr); return FAILED);

    // 5. Get the output value, copy the result from device memory to host memory
    std::vector<T> resultData(size);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outputZDeviceAddr, size * sizeof(T),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyTensors(tensors, deviceAddrs, workspaceAddr); return FAILED);

    // 6. Destroy resources created in this case
    DestroyTensors(tensors, deviceAddrs, workspaceAddr);

    // print the output result
    // 期望结果：z = x / y = 1.0 / 2.0 = 0.5
    std::vector<T> goldenData(size, toHostValue(0.5));

    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("%.1f ", toFloatValue(resultData[i]));
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

int main(int argc, char **argv)
{
    // 1. (Fixed code) Initialize device / stream, refer to the list of external interfaces of acl
    // Update deviceId to your own device id
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. Create input and output, need to customize according to the interface of the API
    std::vector<int64_t> shape = {8, 2048};

    // float16 用例（与算子原型声明的 float16 支持一致）
    ret = RunDivCase<aclFloat16>(stream, aclDataType::ACL_FLOAT16, shape,
                                 [](float v) { return aclFloatToFloat16(v); },
                                 [](aclFloat16 v) { return aclFloat16ToFloat(v); });
    CHECK_RET(ret == SUCCESS, LOG_PRINT("float16 case failed. ERROR: %d\n", ret);
              aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return FAILED);

    // float32 用例（与算子原型声明的 float32 支持一致）
    ret = RunDivCase<float>(stream, aclDataType::ACL_FLOAT, shape, [](float v) { return v; },
                            [](float v) { return v; });
    CHECK_RET(ret == SUCCESS, LOG_PRINT("float32 case failed. ERROR: %d\n", ret);
              aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return FAILED);

    // 7. Destroy stream, reset device and finalize acl
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
