/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ... (License header)
 */

/*!
 * \file test_aclnn_relu.cpp
 * \brief ReLU ACLNN 调用示例
 *
 * 4 维 shape 测试：负值 → 0, 正值 → 不变, 零 → 0
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <half/half.hpp>
#include "acl/acl.h"
#include "aclnn/aclnn_relu.h"

using half_float::half;

struct TestCase {
    std::string name;
    std::vector<int64_t> shape;
    int64_t numElements;
};

static const std::vector<TestCase> TEST_CASES = {
    {"[1,16,16,16]",   {1, 16, 16, 16},   4096},
    {"[4,16,32,32]",   {4, 16, 32, 32},   65536},
    {"[8,32,64,64]",   {8, 32, 64, 64},   1048576},
};

static int g_failed = 0;

static float cpuRelu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

static bool runTestCase(aclrtStream stream, const TestCase& tc)
{
    int64_t numElems = tc.numElements;
    int64_t dataSize = numElems * sizeof(half);

    // 输入数据：负值、正值、零、边界
    std::vector<half> inputData(numElems);
    for (int64_t i = 0; i < numElems; i++) {
        float val;
        if (i < numElems / 3)      val = -static_cast<float>(i + 1);        // 负值
        else if (i < 2 * numElems / 3) val = static_cast<float>(i % 100);  // 正值
        else                         val = 0.0f;                           // 零
        // 特殊值
        if (i == 0) val = -65504.0f;    // -max
        if (i == 1) val = 65504.0f;     // +max
        if (i == 2) val = -0.0f;        // 负零
        inputData[i] = half(val);
    }

    // CPU Golden
    std::vector<half> expected(numElems);
    for (int64_t i = 0; i < numElems; i++) {
        expected[i] = half(cpuRelu(static_cast<float>(inputData[i])));
    }

    // Device
    half *deviceX = nullptr, *deviceY = nullptr;
    aclrtMalloc((void**)&deviceX, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&deviceY, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(deviceX, dataSize, inputData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensorDesc* xDesc = aclCreateTensorDesc(ACL_FLOAT16, tc.shape.size(), tc.shape.data(), ACL_FORMAT_ND);
    aclTensorDesc* yDesc = aclCreateTensorDesc(ACL_FLOAT16, tc.shape.size(), tc.shape.data(), ACL_FORMAT_ND);
    aclSetTensorShape(yDesc, tc.shape.size(), tc.shape.data());

    aclTensor* xTensor = aclCreateTensor(xDesc, deviceX, dataSize, ACL_MEM_MALLOC_HUGE_FIRST, nullptr, 0, nullptr);
    aclTensor* yTensor = aclCreateTensor(yDesc, deviceY, dataSize, ACL_MEM_MALLOC_HUGE_FIRST, nullptr, 0, nullptr);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnReluGetWorkspaceSize(xTensor, yTensor, &wsSize, &executor);
    void* ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnRelu(ws, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    std::vector<half> output(numElems);
    aclrtMemcpy(output.data(), dataSize, deviceY, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    constexpr float RTOL = 1e-3f, ATOL = 1e-6f;
    int mismatch = 0;
    for (int64_t i = 0; i < numElems && mismatch < 5; i++) {
        float cpuVal = static_cast<float>(expected[i]);
        float devVal = static_cast<float>(output[i]);
        float diff = std::abs(cpuVal - devVal);
        float maxVal = std::max(std::abs(cpuVal), std::abs(devVal));
        if (diff > ATOL && (maxVal < 1e-6f || diff / maxVal > RTOL)) {
            std::cerr << "  MISMATCH [" << i << "]: cpu=" << cpuVal << " dev=" << devVal << std::endl;
            mismatch++;
        }
    }

    aclDestroyTensor(xTensor); aclDestroyTensor(yTensor);
    aclDestroyTensorDesc(xDesc); aclDestroyTensorDesc(yDesc);
    aclrtFree(deviceX); aclrtFree(deviceY);
    if (ws) aclrtFree(ws);

    std::cout << (mismatch == 0 ? "  PASS" : "  FAIL") << std::endl;
    return mismatch == 0;
}

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::cout << "=== ReLU Operator Test ===" << std::endl;
    std::cout << "Data type: float16 | Formula: ReLU(x) = max(0, x)" << std::endl << std::endl;

    for (const auto& tc : TEST_CASES) {
        std::cout << "Shape " << tc.name << " (" << tc.numElements << " elements) ..." << std::endl;
        if (!runTestCase(stream, tc)) g_failed++;
    }

    std::cout << std::endl;
    std::cout << (g_failed == 0 ? "All tests passed!" : std::to_string(g_failed) + " test(s) FAILED!") << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return g_failed;
}
