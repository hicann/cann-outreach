#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

size_t AlignUp(size_t value, size_t align)
{
    return ((value + align - 1) / align) * align;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclInit failed. ERROR: " << ret << std::endl; return FAILED);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cout << "aclrtSetDevice failed. ERROR: " << ret << std::endl;
        aclFinalize();
        return FAILED;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "aclrtCreateStream failed. ERROR: " << ret << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return FAILED;
    }
    return SUCCESS;
}

void Finalize(aclrtStream stream, int32_t deviceId)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
    static aclDataType AclType()
    {
        return ACL_FLOAT;
    }

    static float ToFloat(float value)
    {
        return value;
    }

    static float FromFloat(float value)
    {
        return value;
    }

    static float Tolerance()
    {
        return 1e-5f;
    }

    static const char *Name()
    {
        return "float32";
    }
};

template <>
struct DTypeTraits<aclFloat16> {
    static aclDataType AclType()
    {
        return ACL_FLOAT16;
    }

    static float ToFloat(aclFloat16 value)
    {
        return aclFloat16ToFloat(value);
    }

    static aclFloat16 FromFloat(float value)
    {
        return aclFloatToFloat16(value);
    }

    static float Tolerance()
    {
        return 1e-2f;
    }

    static const char *Name()
    {
        return "float16";
    }
};

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclTensor **tensor)
{
    const auto logicalByteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    const auto alignedByteSize = AlignUp(logicalByteSize, 32);
    const auto alignedElementCount = alignedByteSize / sizeof(T);
    std::vector<T> paddedData(alignedElementCount, DTypeTraits<T>::FromFloat(0.0f));
    std::copy(hostData.begin(), hostData.end(), paddedData.begin());

    auto ret = aclrtMalloc(deviceAddr, alignedByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMalloc failed. ERROR: " << ret << std::endl; return FAILED);

    ret = aclrtMemcpy(*deviceAddr, alignedByteSize, paddedData.data(), alignedByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMemcpy failed. ERROR: " << ret << std::endl; return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), DTypeTraits<T>::AclType(), nullptr, 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, std::cout << "aclCreateTensor failed." << std::endl; return FAILED);
    return SUCCESS;
}

void DestroyCaseResources(aclTensor *inputX, aclTensor *inputY, aclTensor *outputZ,
                          void *inputXDeviceAddr, void *inputYDeviceAddr, void *outputZDeviceAddr,
                          void *workspaceAddr)
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

float MakeXValue(int64_t index)
{
    if (index % 11 == 0) {
        return 0.0f;
    }
    return (static_cast<int32_t>(index % 17) - 8) * 0.25f + 1.2f;
}

float MakeYValue(int64_t index)
{
    if (index % 13 == 0) {
        return 0.0f;
    }
    return (static_cast<int32_t>(index % 19) - 9) * -0.125f + 0.7f;
}

template <typename T>
int RunCase(const std::string &caseName, const std::vector<int64_t> &shape, aclrtStream stream)
{
    const int64_t elementCount = GetShapeSize(shape);
    std::vector<T> inputXHostData(elementCount);
    std::vector<T> inputYHostData(elementCount);
    std::vector<T> outputZHostData(elementCount, DTypeTraits<T>::FromFloat(0.0f));
    std::vector<float> goldenData(elementCount);

    for (int64_t i = 0; i < elementCount; ++i) {
        inputXHostData[i] = DTypeTraits<T>::FromFloat(MakeXValue(i));
        inputYHostData[i] = DTypeTraits<T>::FromFloat(MakeYValue(i));
        goldenData[i] = DTypeTraits<T>::ToFloat(inputXHostData[i]) + DTypeTraits<T>::ToFloat(inputYHostData[i]);
    }

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, &inputX);
    CHECK_RET(ret == SUCCESS,
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, &inputY);
    CHECK_RET(ret == SUCCESS,
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, &outputZ);
    CHECK_RET(ret == SUCCESS,
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnAddCustomTemplateGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              std::cout << "aclnnAddCustomTemplateGetWorkspaceSize failed. ERROR: " << ret << std::endl;
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  std::cout << "allocate workspace failed. ERROR: " << ret << std::endl;
                  DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                       workspaceAddr);
                  return FAILED);
    }

    ret = aclnnAddCustomTemplate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              std::cout << "aclnnAddCustomTemplate failed. ERROR: " << ret << std::endl;
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              std::cout << "aclrtSynchronizeStream failed. ERROR: " << ret << std::endl;
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    std::vector<T> resultData(elementCount);
    const auto byteSize = static_cast<size_t>(elementCount) * sizeof(T);
    ret = aclrtMemcpy(resultData.data(), byteSize, outputZDeviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              std::cout << "copy result from device to host failed. ERROR: " << ret << std::endl;
              DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                   workspaceAddr);
              return FAILED);

    const int64_t maxPrintSize = std::min<int64_t>(elementCount, 10);
    std::cout << "Running case: " << DTypeTraits<T>::Name() << " " << caseName
              << ", elements = " << elementCount << std::endl;
    std::cout << "Output: ";
    for (int64_t i = 0; i < maxPrintSize; ++i) {
        std::cout << DTypeTraits<T>::ToFloat(resultData[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Golden: ";
    for (int64_t i = 0; i < maxPrintSize; ++i) {
        std::cout << goldenData[i] << " ";
    }
    std::cout << std::endl;

    const float tolerance = DTypeTraits<T>::Tolerance();
    for (int64_t i = 0; i < elementCount; ++i) {
        const float output = DTypeTraits<T>::ToFloat(resultData[i]);
        const float diff = std::abs(output - goldenData[i]);
        if (diff > tolerance) {
            std::cout << "[Failed] mismatch at index " << i << ", output is " << output
                      << ", golden is " << goldenData[i] << std::endl;
            DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                                 workspaceAddr);
            return FAILED;
        }
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    DestroyCaseResources(inputX, inputY, outputZ, inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr,
                         workspaceAddr);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, std::cout << "Init acl failed. ERROR: " << ret << std::endl; return FAILED);

    struct CaseDef {
        std::string name;
        std::vector<int64_t> shape;
    };
    const std::vector<CaseDef> cases = {
        {"lesson_shape", {45, 20480}},
        {"scalar", {1}},
        {"small_unaligned", {7}},
        {"one_32b_block_float32", {8}},
        {"unaligned_tail", {13}},
        {"medium_uneven_core", {257}},
        {"large_unaligned", {16387}},
    };

    for (const auto &caseDef : cases) {
        if (RunCase<float>(caseDef.name, caseDef.shape, stream) != SUCCESS) {
            Finalize(stream, deviceId);
            return FAILED;
        }
        if (RunCase<aclFloat16>(caseDef.name, caseDef.shape, stream) != SUCCESS) {
            Finalize(stream, deviceId);
            return FAILED;
        }
    }

    Finalize(stream, deviceId);
    return SUCCESS;
}
