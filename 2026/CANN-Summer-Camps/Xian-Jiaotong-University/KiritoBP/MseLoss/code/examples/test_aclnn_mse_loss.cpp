// test_aclnn_mse_loss.cpp
// MseLoss 算子 ACLNN 调用测试程序
// 参考: @wxy19931018 TruncateMod examples/test_aclnn_truncate_mod.cpp

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_mse_loss.h"

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

// float → float16 转换
static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
            elemSize = 2;
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
    } else {
        memcpy(convBuf.data(), hostData.data(), size);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入 predict / label
    std::vector<int64_t> shape = {8, 2048};
    int64_t elemCount = GetShapeSize(shape);

    aclTensor* predict = nullptr;
    aclTensor* label = nullptr;
    aclTensor* y = nullptr;
    void *pDev = nullptr, *lDev = nullptr, *yDev = nullptr;

    std::vector<float> pHost(elemCount, 1.0f);
    std::vector<float> lHost(elemCount, 2.0f);
    std::vector<float> yHost(elemCount, 0.0f);

    ret = CreateAclTensor(pHost, shape, &pDev, aclDataType::ACL_FLOAT, &predict);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lHost, shape, &lDev, aclDataType::ACL_FLOAT, &label);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHost, shape, &yDev, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnMseLossGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMseLossGetWorkspaceSize(predict, label, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMseLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnMseLoss
    ret = aclnnMseLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMseLoss failed. ERROR: %d\n", ret); return ret);

    // 6. 同步
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 读取输出并验证
    std::vector<float> result(elemCount, 0);
    ret = aclrtMemcpy(result.data(), elemCount * sizeof(float), yDev,
                      elemCount * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    // 理论值: (1.0 - 2.0)^2 = 1.0
    std::vector<float> golden(elemCount, 1.0f);

    LOG_PRINT("result (first 10): ");
    for (int i = 0; i < 10; i++) LOG_PRINT("%.4f ", result[i]);
    LOG_PRINT("\n");

    if (std::equal(result.begin(), result.end(), golden.begin(),
                   [](float a, float b) { return std::fabs(a - b) < 1e-3; })) {
        LOG_PRINT("test pass\n");
    } else {
        LOG_PRINT("test failed\n");
        return 1;
    }

    // 8. 释放资源
    aclDestroyTensor(predict); aclrtFree(pDev);
    aclDestroyTensor(label);   aclrtFree(lDev);
    aclDestroyTensor(y);       aclrtFree(yDev);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    if (executor) aclDestroyOpExecutor(executor);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
