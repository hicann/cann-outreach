#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sub_custom_template.h"

namespace {
constexpr int32_t kDeviceId = 0;
constexpr int64_t kRows = 8;
constexpr int64_t kCols = 2048;
constexpr size_t kTotalLength = kRows * kCols;
constexpr float kValueX = 1.0f;
constexpr float kValueY = 2.0f;
constexpr float kGoldenValue = kValueX - kValueY;

bool CheckAcl(aclError ret, const char *expr)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << expr << " failed, ret = " << ret << std::endl;
        return false;
    }
    return true;
}

bool CheckAclnn(aclnnStatus ret, const char *expr)
{
    if (ret != 0) {
        std::cerr << expr << " failed, ret = " << ret << std::endl;
        return false;
    }
    return true;
}

bool PrintAndVerify(const std::vector<float> &output)
{
    std::cout << "result is:" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < 10; ++i) {
        std::cout << output[i] << (i + 1 == 10 ? '\n' : ' ');
    }

    const bool ok = std::all_of(output.begin(), output.end(), [](float value) {
        return std::fabs(value - kGoldenValue) < 1e-5f;
    });
    std::cout << (ok ? "test pass" : "test failed") << std::endl;
    return ok;
}
} // namespace

int main()
{
    std::vector<float> x(kTotalLength, kValueX);
    std::vector<float> y(kTotalLength, kValueY);
    std::vector<float> z(kTotalLength, 0.0f);

    const int64_t shape[] = {kRows, kCols};
    const int64_t strides[] = {kCols, 1};
    const size_t bytes = kTotalLength * sizeof(float);

    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret = " << initRet
                  << ". Skip device execution on this host." << std::endl;
        return 0;
    }

    void *devX = nullptr;
    void *devY = nullptr;
    void *devZ = nullptr;
    void *workspace = nullptr;
    aclrtStream stream = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *yTensor = nullptr;
    aclTensor *zTensor = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    ok = ok && CheckAcl(aclrtSetDevice(kDeviceId), "aclrtSetDevice");
    ok = ok && CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    ok = ok && CheckAcl(aclrtMalloc(&devX, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc devX");
    ok = ok && CheckAcl(aclrtMalloc(&devY, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc devY");
    ok = ok && CheckAcl(aclrtMalloc(&devZ, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc devZ");
    ok = ok && CheckAcl(aclrtMemcpy(devX, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy x");
    ok = ok && CheckAcl(aclrtMemcpy(devY, bytes, y.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy y");

    if (ok) {
        xTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devX);
        yTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devY);
        zTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devZ);
        ok = (xTensor != nullptr && yTensor != nullptr && zTensor != nullptr);
    }

    ok = ok && CheckAclnn(aclnnSubCustomTemplateGetWorkspaceSize(
        xTensor, yTensor, zTensor, &workspaceSize, &executor), "aclnnSubCustomTemplateGetWorkspaceSize");
    if (ok && workspaceSize > 0) {
        ok = CheckAcl(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }
    ok = ok && CheckAclnn(aclnnSubCustomTemplate(workspace, workspaceSize, executor, stream), "aclnnSubCustomTemplate");
    ok = ok && CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    ok = ok && CheckAcl(aclrtMemcpy(z.data(), bytes, devZ, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy z");

    if (zTensor != nullptr) {
        aclDestroyTensor(zTensor);
    }
    if (yTensor != nullptr) {
        aclDestroyTensor(yTensor);
    }
    if (xTensor != nullptr) {
        aclDestroyTensor(xTensor);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (devZ != nullptr) {
        aclrtFree(devZ);
    }
    if (devY != nullptr) {
        aclrtFree(devY);
    }
    if (devX != nullptr) {
        aclrtFree(devX);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    return ok && PrintAndVerify(z) ? 0 : 1;
}
