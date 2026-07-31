#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_softshrink_grad.h"

namespace {
constexpr int SUCCESS = 0;
constexpr int FAILED = 1;

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

bool CheckAcl(aclError error, const char *operation)
{
    if (error != ACL_SUCCESS) {
        std::printf("%s failed, error=%d\n", operation, error);
        return false;
    }
    return true;
}

bool CreateTensor(const std::vector<float> &hostData,
                  const std::vector<int64_t> &shape,
                  void **deviceAddress, aclTensor **tensor)
{
    const size_t bytes = hostData.size() * sizeof(float);
    if (!CheckAcl(aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc")) {
        return false;
    }
    if (!CheckAcl(aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy H2D")) {
        return false;
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(),
                              *deviceAddress);
    return *tensor != nullptr;
}

void ReleaseTensor(aclTensor *tensor, void *address)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (address != nullptr) {
        aclrtFree(address);
    }
}
}  // namespace

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        return FAILED;
    }

    const std::vector<int64_t> shape{2, 8};
    const float lambd = 0.5F;
    const std::vector<float> gradOutput{
        1, 2, 3, 4, 5, 6, 7, 8,
        -1, -2, -3, -4, -5, -6, -7, -8};
    const std::vector<float> self{
        -1.0F, -0.5F, -0.49F, 0.0F, 0.49F, 0.5F, 0.51F, 2.0F,
        -2.0F, -0.51F, -0.5F, -0.1F, 0.1F, 0.5F, 0.51F, 3.0F};
    std::vector<float> output(ShapeSize(shape), 0.0F);
    std::vector<float> golden(output.size(), 0.0F);
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] = std::fabs(self[i]) > lambd ? gradOutput[i] : 0.0F;
    }

    void *gradAddress = nullptr;
    void *selfAddress = nullptr;
    void *outputAddress = nullptr;
    aclTensor *gradTensor = nullptr;
    aclTensor *selfTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    if (!CreateTensor(gradOutput, shape, &gradAddress, &gradTensor) ||
        !CreateTensor(self, shape, &selfAddress, &selfTensor) ||
        !CreateTensor(output, shape, &outputAddress, &outputTensor)) {
        ReleaseTensor(gradTensor, gradAddress);
        ReleaseTensor(selfTensor, selfAddress);
        ReleaseTensor(outputTensor, outputAddress);
        return FAILED;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclError result = aclnnSoftshrinkGradGetWorkspaceSize(
        gradTensor, selfTensor, lambd, outputTensor, &workspaceSize, &executor);
    if (!CheckAcl(result, "aclnnSoftshrinkGradGetWorkspaceSize")) {
        return FAILED;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0 &&
        !CheckAcl(aclrtMalloc(&workspace, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "workspace allocation")) {
        return FAILED;
    }
    result = aclnnSoftshrinkGrad(workspace, workspaceSize, executor, stream);
    if (!CheckAcl(result, "aclnnSoftshrinkGrad") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size() * sizeof(float),
                              outputAddress, output.size() * sizeof(float),
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy D2H")) {
        return FAILED;
    }

    bool passed = true;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - golden[i]) > 1e-6F) {
            std::printf("mismatch at %zu: actual=%f expected=%f\n",
                        i, output[i], golden[i]);
            passed = false;
        }
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    ReleaseTensor(gradTensor, gradAddress);
    ReleaseTensor(selfTensor, selfAddress);
    ReleaseTensor(outputTensor, outputAddress);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::printf("SoftshrinkGrad ACLNN test: %s\n", passed ? "PASS" : "FAIL");
    return passed ? SUCCESS : FAILED;
}
