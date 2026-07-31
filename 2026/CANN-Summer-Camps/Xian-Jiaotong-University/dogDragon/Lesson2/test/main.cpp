/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

void ReleaseTensorResources(aclTensor *inputX, aclTensor *inputY, aclTensor *outputZ, void *inputXAddr,
                            void *inputYAddr, void *outputZAddr, void *workspaceAddr)
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
    if (inputXAddr != nullptr) {
        aclrtFree(inputXAddr);
    }
    if (inputYAddr != nullptr) {
        aclrtFree(inputYAddr);
    }
    if (outputZAddr != nullptr) {
        aclrtFree(outputZAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
}

template <typename T>
struct TestTypeTraits;

template <>
struct TestTypeTraits<aclFloat16> {
    static aclDataType DataType()
    {
        return aclDataType::ACL_FLOAT16;
    }
    static aclFloat16 FromFloat(float value)
    {
        return aclFloatToFloat16(value);
    }
    static float ToFloat(aclFloat16 value)
    {
        return aclFloat16ToFloat(value);
    }
    static const char *Name()
    {
        return "float16";
    }
    static float Tolerance()
    {
        return 1e-3F;
    }
};

template <>
struct TestTypeTraits<float> {
    static aclDataType DataType()
    {
        return aclDataType::ACL_FLOAT;
    }
    static float FromFloat(float value)
    {
        return value;
    }
    static float ToFloat(float value)
    {
        return value;
    }
    static const char *Name()
    {
        return "float32";
    }
    static float Tolerance()
    {
        return 1e-6F;
    }
};

template <typename T>
int RunTest(aclrtStream stream)
{
    const std::vector<int64_t> shape = {8, 2048};
    const int64_t elementCount = GetShapeSize(shape);

    std::vector<T> inputXHostData(elementCount);
    std::vector<T> inputYHostData(elementCount);
    std::vector<T> outputZHostData(elementCount, TestTypeTraits<T>::FromFloat(0.0F));
    std::vector<T> goldenData(elementCount);
    for (int64_t i = 0; i < elementCount; ++i) {
        const float x = 1.0F + static_cast<float>(i % 17) * 0.25F;
        const float y = 0.5F + static_cast<float>(i % 13) * 0.125F;
        inputXHostData[i] = TestTypeTraits<T>::FromFloat(x);
        inputYHostData[i] = TestTypeTraits<T>::FromFloat(y);
        goldenData[i] = TestTypeTraits<T>::FromFloat(TestTypeTraits<T>::ToFloat(inputXHostData[i]) -
                                                     TestTypeTraits<T>::ToFloat(inputYHostData[i]));
    }

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    auto cleanup = [&]() {
        ReleaseTensorResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                               workspaceAddr);
    };

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, TestTypeTraits<T>::DataType(), &inputX);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, TestTypeTraits<T>::DataType(), &inputY);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);
    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, TestTypeTraits<T>::DataType(), &outputZ);
    CHECK_RET(ret == SUCCESS, cleanup(); return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSubCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s workspace query failed. ERROR: %d\n", TestTypeTraits<T>::Name(), ret);
              cleanup(); return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("%s workspace allocation failed. ERROR: %d\n", TestTypeTraits<T>::Name(), ret);
                  cleanup(); return FAILED);
    }

    ret = aclnnSubCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s operator execution failed. ERROR: %d\n", TestTypeTraits<T>::Name(), ret);
              cleanup(); return FAILED);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s stream synchronization failed. ERROR: %d\n", TestTypeTraits<T>::Name(), ret);
              cleanup(); return FAILED);

    std::vector<T> resultData(elementCount);
    const auto outputSize = elementCount * sizeof(T);
    ret = aclrtMemcpy(resultData.data(), outputSize, outputZDeviceAddr, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s result copy failed. ERROR: %d\n", TestTypeTraits<T>::Name(), ret);
              cleanup(); return FAILED);

    bool passed = true;
    for (int64_t i = 0; i < elementCount; ++i) {
        const float actual = TestTypeTraits<T>::ToFloat(resultData[i]);
        const float expected = TestTypeTraits<T>::ToFloat(goldenData[i]);
        if (std::fabs(actual - expected) > TestTypeTraits<T>::Tolerance()) {
            LOG_PRINT("%s mismatch at %lld: expected %.6f, actual %.6f\n", TestTypeTraits<T>::Name(),
                      static_cast<long long>(i), expected, actual);
            passed = false;
            break;
        }
    }

    LOG_PRINT("%s first 10 results: ", TestTypeTraits<T>::Name());
    for (int64_t i = 0; i < 10; ++i) {
        LOG_PRINT("%.3f ", TestTypeTraits<T>::ToFloat(resultData[i]));
    }
    LOG_PRINT("\n%s test %s\n", TestTypeTraits<T>::Name(), passed ? "pass" : "failed");
    cleanup();
    return passed ? SUCCESS : FAILED;
}

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    ret = RunTest<aclFloat16>(stream);
    if (ret == SUCCESS) {
        ret = RunTest<float>(stream);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
