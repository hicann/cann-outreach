/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_relu_example_tiling.cpp
 * \brief ReluExample Tiling 单元测试
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_kernel/arch22/relu_example_tiling_data.h"

class ReluExampleTilingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ReluExampleTilingTest, DefaultValues) {
    ReluExampleTilingData tiling;
    EXPECT_EQ(tiling.totalNum, 0);
    EXPECT_EQ(tiling.blockFactor, 0);
    EXPECT_EQ(tiling.ubFactor, 0);
}

TEST_F(ReluExampleTilingTest, Shape4D_Small) {
    // [N4,N3,N2,N1] = [1,1,1,128] → 128 elements
    ReluExampleTilingData tiling;
    tiling.totalNum = 1 * 1 * 1 * 128;
    tiling.blockFactor = 128;
    EXPECT_EQ(tiling.totalNum, 128);
    EXPECT_GT(tiling.blockFactor, 0);
}

TEST_F(ReluExampleTilingTest, Shape4D_Medium) {
    // [N4,N3,N2,N1] = [4,8,16,32] → 16384 elements
    ReluExampleTilingData tiling;
    tiling.totalNum = 4 * 8 * 16 * 32;
    tiling.blockFactor = 4096;
    tiling.ubFactor = 256;
    EXPECT_EQ(tiling.totalNum, 16384);
    EXPECT_EQ(tiling.blockFactor, 4096);
    EXPECT_GT(tiling.ubFactor, 0);
    EXPECT_LE(tiling.ubFactor, tiling.blockFactor);
}

TEST_F(ReluExampleTilingTest, Shape4D_Large) {
    // [N4,N3,N2,N1] = [128,64,32,16] → 4194304 elements
    ReluExampleTilingData tiling;
    tiling.totalNum = 128 * 64 * 32 * 16;
    tiling.blockFactor = 524288;
    tiling.ubFactor = 1024;
    EXPECT_EQ(tiling.totalNum, 4194304);
    EXPECT_GT(tiling.blockFactor, 0);
    EXPECT_GT(tiling.ubFactor, 0);
}

TEST_F(ReluExampleTilingTest, UbFactorLimit) {
    constexpr int64_t MAX_UB_ELEMENTS = 256 * 1024 / 2;  // 256KB UB / 2B per half
    ReluExampleTilingData tiling;
    tiling.ubFactor = MAX_UB_ELEMENTS;
    EXPECT_LE(tiling.ubFactor * static_cast<int64_t>(sizeof(half)),
              static_cast<int64_t>(256 * 1024));
}
