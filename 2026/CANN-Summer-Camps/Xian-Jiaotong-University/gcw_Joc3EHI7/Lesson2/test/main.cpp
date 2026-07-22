#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sub_custom_template.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
const std::vector<std::vector<int64_t>> SHAPES = {
    {8, 2048},
    {8, 2050},
};

template <typename T>
T FromFloat(float value);

template <>
float FromFloat<float>(float value)
{
    return value;
}

template <>
aclFloat16 FromFloat<aclFloat16>(float value)
{
    return aclFloatToFloat16(value);
}

template <typename T>
float ToFloat(T value);

template <>
float ToFloat<float>(float value)
{
    return value;
}

template <>
float ToFloat<aclFloat16>(aclFloat16 value)
{
    return aclFloat16ToFloat(value);
}

int64_t GetElementCount(const std::vector<int64_t> &shape)
{
    int64_t elementCount = 1;
    for (int64_t dim : shape) {
        elementCount *= dim;
    }
    return elementCount;
}

template <typename T>
bool RunCase(aclrtStream stream, aclDataType dataType, const std::vector<int64_t> &shape,
             const std::string &name, float tolerance)
{
    const int64_t elementCount = GetElementCount(shape);
    const size_t byteSize = static_cast<size_t>(elementCount) * sizeof(T);
    std::vector<T> xHost(elementCount);
    std::vector<T> yHost(elementCount);
    std::vector<T> zHost(elementCount, FromFloat<T>(0.0F));
    for (int64_t i = 0; i < elementCount; ++i) {
        xHost[i] = FromFloat<T>(static_cast<float>(i % 31) * 0.5F);
        yHost[i] = FromFloat<T>(static_cast<float>(i % 7) * 0.25F);
    }

    void *xDevice = nullptr;
    void *yDevice = nullptr;
    void *zDevice = nullptr;
    void *workspace = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *yTensor = nullptr;
    aclTensor *zTensor = nullptr;
    aclOpExecutor *executor = nullptr;

    auto cleanup = [&]() {
        if (xTensor != nullptr) aclDestroyTensor(xTensor);
        if (yTensor != nullptr) aclDestroyTensor(yTensor);
        if (zTensor != nullptr) aclDestroyTensor(zTensor);
        if (xDevice != nullptr) aclrtFree(xDevice);
        if (yDevice != nullptr) aclrtFree(yDevice);
        if (zDevice != nullptr) aclrtFree(zDevice);
        if (workspace != nullptr) aclrtFree(workspace);
    };

    aclError ret = aclrtMalloc(&xDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) ret = aclrtMalloc(&yDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) ret = aclrtMalloc(&zDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) ret = aclrtMemcpy(xDevice, byteSize, xHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret == ACL_SUCCESS) ret = aclrtMemcpy(yDevice, byteSize, yHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << name << ": device allocation/copy failed, ret=" << ret << '\n';
        cleanup();
        return false;
    }

    xTensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), xDevice);
    yTensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), yDevice);
    zTensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), zDevice);
    if (xTensor == nullptr || yTensor == nullptr || zTensor == nullptr) {
        std::cerr << name << ": aclCreateTensor failed\n";
        cleanup();
        return false;
    }

    uint64_t workspaceSize = 0;
    ret = aclnnSubCustomTemplateGetWorkspaceSize(xTensor, yTensor, zTensor, &workspaceSize, &executor);
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnSubCustomTemplate(workspace, workspaceSize, executor, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(zHost.data(), byteSize, zDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACL_SUCCESS) {
        std::cerr << name << ": operator execution failed, ret=" << ret << '\n';
        cleanup();
        return false;
    }

    float maxError = 0.0F;
    int64_t mismatch = -1;
    for (int64_t i = 0; i < elementCount; ++i) {
        const float expected = ToFloat(xHost[i]) - ToFloat(yHost[i]);
        const float error = std::fabs(ToFloat(zHost[i]) - expected);
        if (error > maxError) maxError = error;
        if (error > tolerance && mismatch < 0) mismatch = i;
    }

    if (mismatch >= 0) {
        std::cerr << name << ": FAILED at index " << mismatch
                  << ", actual=" << ToFloat(zHost[mismatch])
                  << ", expected=" << (ToFloat(xHost[mismatch]) - ToFloat(yHost[mismatch]))
                  << ", max_error=" << maxError << '\n';
        cleanup();
        return false;
    }

    std::cout << name << ": PASS, shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << shape[i];
    }
    std::cout << "], elements=" << elementCount << ", max_error=" << maxError << '\n';
    cleanup();
    return true;
}
} // namespace

int main()
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret=" << ret << '\n';
        return 1;
    }
    ret = aclrtSetDevice(DEVICE_ID);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, ret=" << ret << '\n';
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed, ret=" << ret << '\n';
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
        return 1;
    }

    bool fp16Passed = true;
    bool fp32Passed = true;
    for (const auto &shape : SHAPES) {
        fp16Passed = RunCase<aclFloat16>(stream, ACL_FLOAT16, shape, "FP16", 1.0e-3F) && fp16Passed;
        fp32Passed = RunCase<float>(stream, ACL_FLOAT, shape, "FP32", 1.0e-6F) && fp32Passed;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return fp16Passed && fp32Passed ? 0 : 1;
}
