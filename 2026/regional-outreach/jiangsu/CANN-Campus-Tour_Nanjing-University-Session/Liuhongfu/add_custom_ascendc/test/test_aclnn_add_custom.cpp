/**
 * @file test_aclnn_add_custom.cpp
 * @brief add_custom 算子的 aclnn 两段式接口调用样例 + 精度验证。
 *
 * 编译：见 scripts/build.sh
 * 运行：./test_aclnn_add_custom [deviceId] [N2] [N1]
 *       默认 deviceId=0, N2=8, N1=257（刻意取非 32B 对齐的列数，用于覆盖尾块逻辑）
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "aclnn_add_custom.h"

#include "utils.h"

#define CHECK_ACL(expr, msg)                                                      \
    do {                                                                          \
        const aclError _ret = (expr);                                             \
        if (_ret != ACL_SUCCESS) {                                                \
            std::printf("[ERROR] %s failed, aclError = %d\n", (msg),              \
                        static_cast<int32_t>(_ret));                              \
            return -1;                                                            \
        }                                                                         \
    } while (0)

#define CHECK_ACLNN(expr, msg)                                                    \
    do {                                                                          \
        const aclnnStatus _ret = (expr);                                          \
        if (_ret != ACLNN_SUCCESS) {                                              \
            std::printf("[ERROR] %s failed, aclnnStatus = %d\n", (msg),           \
                        static_cast<int32_t>(_ret));                              \
            return -1;                                                            \
        }                                                                         \
    } while (0)

int main(int argc, char** argv)
{
    int32_t deviceId = 0;
    int64_t n2 = 8;
    int64_t n1 = 257;
    if (argc > 1) { deviceId = std::atoi(argv[1]); }
    if (argc > 2) { n2 = std::atoll(argv[2]); }
    if (argc > 3) { n1 = std::atoll(argv[3]); }

    const int64_t elementCount = n2 * n1;
    const size_t byteSize = static_cast<size_t>(elementCount) * sizeof(uint16_t);
    std::printf("== aclnnAddCustom test: device=%d, shape=[%ld, %ld], elements=%ld ==\n",
                deviceId, static_cast<long>(n2), static_cast<long>(n1),
                static_cast<long>(elementCount));

    // ---- 1. 构造 host 侧输入与参考结果 ------------------------------------
    std::vector<uint16_t> xHost(static_cast<size_t>(elementCount));
    std::vector<uint16_t> yHost(static_cast<size_t>(elementCount));
    std::vector<uint16_t> zExpect(static_cast<size_t>(elementCount));
    for (int64_t i = 0; i < elementCount; ++i) {
        // 取值控制在 fp16 能精确表示的范围内，避免输入本身就带舍入误差
        const float xv = static_cast<float>((i % 97)) * 0.125F - 5.0F;
        const float yv = static_cast<float>((i % 31)) * 0.25F - 2.0F;
        xHost[static_cast<size_t>(i)] = FloatToFp16(xv);
        yHost[static_cast<size_t>(i)] = FloatToFp16(yv);
        zExpect[static_cast<size_t>(i)] = FloatToFp16(xv + yv);
    }

    // ---- 2. Init ACL ------------------------------------------------------
    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(deviceId), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    // ---- 3. 申请 device 内存并拷贝输入 -------------------------------------
    void* xDevice = nullptr;
    void* yDevice = nullptr;
    void* zDevice = nullptr;
    CHECK_ACL(aclrtMalloc(&xDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x");
    CHECK_ACL(aclrtMalloc(&yDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc y");
    CHECK_ACL(aclrtMalloc(&zDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc z");

    CHECK_ACL(aclrtMemcpy(xDevice, byteSize, xHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy x");
    CHECK_ACL(aclrtMemcpy(yDevice, byteSize, yHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy y");

    // ---- 4. 构造 aclTensor ------------------------------------------------
    std::vector<int64_t> shape = {n2, n1};
    aclTensor* xTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16,
                                         nullptr, 0, ACL_FORMAT_ND,
                                         shape.data(), shape.size(), xDevice);
    aclTensor* yTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16,
                                         nullptr, 0, ACL_FORMAT_ND,
                                         shape.data(), shape.size(), yDevice);
    aclTensor* zTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16,
                                         nullptr, 0, ACL_FORMAT_ND,
                                         shape.data(), shape.size(), zDevice);
    if (xTensor == nullptr || yTensor == nullptr || zTensor == nullptr) {
        std::printf("[ERROR] aclCreateTensor failed\n");
        return -1;
    }

    // ---- 5. 第一段接口 ----------------------------------------------------
    uint64_t workspaceSize = 0U;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnAddCustomGetWorkspaceSize(xTensor, yTensor, zTensor, &workspaceSize, &executor),
                "aclnnAddCustomGetWorkspaceSize");
    std::printf("workspaceSize = %lu bytes\n", static_cast<unsigned long>(workspaceSize));

    void* workspace = nullptr;
    if (workspaceSize > 0U) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc workspace");
    }

    // ---- 6. 第二段接口 ----------------------------------------------------
    CHECK_ACLNN(aclnnAddCustom(workspace, workspaceSize, executor, stream), "aclnnAddCustom");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    // ---- 7. 回拷并比对 ----------------------------------------------------
    std::vector<uint16_t> zHost(static_cast<size_t>(elementCount));
    CHECK_ACL(aclrtMemcpy(zHost.data(), byteSize, zDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy z");

    size_t mismatchCount = 0U;
    const double maxErr = CompareFp16(zHost, zExpect, mismatchCount);

    std::printf("--------------------------------------------------\n");
    std::printf("elements      : %ld\n", static_cast<long>(elementCount));
    std::printf("mismatch      : %zu\n", mismatchCount);
    std::printf("max abs error : %.6f\n", maxErr);
    std::printf("result        : %s\n", (mismatchCount == 0U) ? "PASS" : "FAIL");
    std::printf("--------------------------------------------------\n");

    // ---- 8. 释放资源 ------------------------------------------------------
    if (workspace != nullptr) { aclrtFree(workspace); }
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(zTensor);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (mismatchCount == 0U) ? 0 : 1;
}
