/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS REQUIRED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_relu_example_st.cpp
 * \brief ReluExample 算子系统测试（ST）
 *
 * Real 模式在 NPU 上运行，验证精度。
 * 覆盖 4D shape [N4,N3,N2,N1] 的各种组合。
 * 数据范围：包含正负数、零、边界值
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include "aclnn_relu_example.h"

#define CHECK_ACL(expr) do { \
    aclError __ret = (expr); \
    ASSERT_EQ(__ret, ACL_SUCCESS) << "ACL error at line " << __LINE__; \
} while(0)

constexpr float RELATIVE_TOL = 1e-3f;
constexpr float ABSOLUTE_TOL = 1e-3f;

int64_t ShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto d : shape) size *= d;
    return size;
}

// 生成 Relu 测试数据（正负数混合）
std::vector<half> GenerateTestData(int64_t numElements) {
    std::vector<half> data(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        // 均匀覆盖正负区间
        float val = -5.0f + (static_cast<float>(i % 997) / 997.0f) * 10.0f;
        // 包含若干特殊值
        if (i % 7 == 0) val = 0.0f;
        if (i % 11 == 0) val = 1.0f;
        if (i % 13 == 0) val = -1.0f;
        if (i % 17 == 0) val = 100.0f;
        if (i % 19 == 0) val = -100.0f;
        data[i] = static_cast<half>(val);
    }
    return data;
}

// CPU Golden: y = max(0, x)
std::vector<half> ComputeGolden(const std::vector<half>& input) {
    std::vector<half> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        float val = static_cast<float>(input[i]);
        output[i] = static_cast<half>((val > 0.0f) ? val : 0.0f);
    }
    return output;
}

bool CompareResults(const std::vector<half>& actual, const std::vector<half>& expected) {
    if (actual.size() != expected.size()) return false;

    for (size_t i = 0; i < actual.size(); i++) {
        float a = static_cast<float>(actual[i]);
        float e = static_cast<float>(expected[i]);
        float absErr = std::abs(a - e);
        float relErr = (std::abs(e) > FLT_EPSILON) ? absErr / std::abs(e) : absErr;

        if (relErr > RELATIVE_TOL && absErr > ABSOLUTE_TOL) {
            std::cerr << "[FAIL] idx=" << i << " expected=" << e
                      << " actual=" << a << " absErr=" << absErr
                      << " relErr=" << relErr << std::endl;
            return false;
        }
    }
    return true;
}

class ReluExampleSTTest : public ::testing::Test {
protected:
    aclrtDeviceId deviceId_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        CHECK_ACL(aclInit(nullptr));
        CHECK_ACL(aclrtSetDevice(deviceId_));
        CHECK_ACL(aclrtCreateContext(&context_, deviceId_));
        CHECK_ACL(aclrtCreateStream(&stream_));
    }

    void TearDown() override {
        aclrtDestroyStream(stream_);
        aclrtDestroyContext(context_);
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

    void RunAndVerify(const std::vector<int64_t>& shape) {
        int64_t numElements = ShapeSize(shape);
        auto input = GenerateTestData(numElements);
        auto golden = ComputeGolden(input);

        aclTensor* x = nullptr;
        aclTensor* y = nullptr;
        void* xDev = nullptr;
        void* yDev = nullptr;
        int64_t dataSize = numElements * sizeof(half);

        CHECK_ACL(aclrtMalloc(&xDev, dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(xDev, dataSize, input.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE));

        CHECK_ACL(aclrtMalloc(&yDev, dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtMemset(yDev, dataSize, 0, dataSize);

        x = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16,
            shape.size(), nullptr, 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), xDev);
        y = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16,
            shape.size(), nullptr, 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), yDev);
        ASSERT_NE(x, nullptr);
        ASSERT_NE(y, nullptr);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        CHECK_ACL(aclnnReluExampleGetWorkspaceSize(x, y, &workspaceSize, &executor));

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        CHECK_ACL(aclnnReluExample(workspaceAddr, workspaceSize, executor, stream_));
        CHECK_ACL(aclrtSynchronizeStream(stream_));

        std::vector<half> result(numElements);
        CHECK_ACL(aclrtMemcpy(result.data(), dataSize, yDev, dataSize, ACL_MEMCPY_DEVICE_TO_HOST));

        EXPECT_TRUE(CompareResults(result, golden));

        aclDestroyTensor(x);
        aclDestroyTensor(y);
        aclrtFree(xDev);
        aclrtFree(yDev);
        if (workspaceAddr) aclrtFree(workspaceAddr);
    }
};

// 4D Shape 测试用例
TEST_F(ReluExampleSTTest, Shape4D_1x1x1x128) {
    RunAndVerify({1, 1, 1, 128});
}

TEST_F(ReluExampleSTTest, Shape4D_1x4x16x32) {
    RunAndVerify({1, 4, 16, 32});
}

TEST_F(ReluExampleSTTest, Shape4D_4x8x16x32) {
    RunAndVerify({4, 8, 16, 32});
}

TEST_F(ReluExampleSTTest, Shape4D_2x16x32x64) {
    RunAndVerify({2, 16, 32, 64});
}

TEST_F(ReluExampleSTTest, Shape4D_8x8x32x128) {
    RunAndVerify({8, 8, 32, 128});
}

TEST_F(ReluExampleSTTest, Shape4D_16x16x16x16) {
    RunAndVerify({16, 16, 16, 16});
}

TEST_F(ReluExampleSTTest, Shape4D_128x64x32x16) {
    RunAndVerify({128, 64, 32, 16});
}
