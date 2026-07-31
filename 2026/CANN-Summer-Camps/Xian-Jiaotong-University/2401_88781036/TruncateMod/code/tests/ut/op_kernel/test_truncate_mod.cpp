/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/truncate_mod.cpp"

using namespace std;

constexpr uint32_t smallCoreDataNum = 1024;
constexpr uint32_t bigCoreDataNum = 1040;
constexpr uint32_t smallTailDataNum = 1024;
constexpr uint32_t bigTailDataNum = 1040;
constexpr uint32_t tmpTileDataNum = 4096;
constexpr uint32_t tmpSmallTailDataNum = 2048;
constexpr uint32_t tmpBigTailDataNum = 2080;

extern "C" __global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling);

class TruncateModTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "truncate_mod_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(system(cmd.c_str()), 0);
        ASSERT_EQ(system("chmod -R 755 ./truncate_mod_data/"), 0);
    }
    static void TearDownTestCase() { std::cout << "truncate_mod_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string TruncateModTest::rootPath = "../../../../experimental/";
const std::string TruncateModTest::dataPath = rootPath + "math/truncate_mod/tests/ut/op_kernel/truncate_mod_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(TruncateModTest, test_case_int32_1)
{
    uint32_t blockDim = 1;
    ASSERT_EQ(system("cd ./truncate_mod_data/ && python3 gen_data.py '(1024)' 'int32'"), 0);
    uint32_t dataCount = 1024;
    size_t inputByteSize = dataCount * sizeof(int32_t);

    std::string x1_fileName = "./truncate_mod_data/int32_input_t1_truncate_mod.bin";
    std::string x2_fileName = "./truncate_mod_data/int32_input_t2_truncate_mod.bin";

    uint8_t* x1 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    ASSERT_NE(x1, nullptr);
    uint8_t* x2 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    ASSERT_NE(x2, nullptr);

    ASSERT_TRUE(ReadFile(x1_fileName, inputByteSize, x1, inputByteSize));
    ASSERT_TRUE(ReadFile(x2_fileName, inputByteSize, x2, inputByteSize));

    size_t outputByteSize = dataCount * sizeof(int32_t);
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    ASSERT_NE(y, nullptr);

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceSize));
    ASSERT_NE(workspace, nullptr);
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(TruncateModTilingData)));
    ASSERT_NE(tiling, nullptr);

    TruncateModTilingData* tilingData = reinterpret_cast<TruncateModTilingData*>(tiling);

    tilingData->smallCoreDataNum = smallCoreDataNum;
    tilingData->bigCoreDataNum = bigCoreDataNum;
    tilingData->smallTailDataNum = smallTailDataNum;
    tilingData->bigTailDataNum = bigTailDataNum;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->tmpTileDataNum = tmpTileDataNum;
    tilingData->tmpSmallTailDataNum = tmpSmallTailDataNum;
    tilingData->tmpBigTailDataNum = tmpBigTailDataNum;
    tilingData->totalDataNum = dataCount;
    tilingData->x1DataNum = dataCount;
    tilingData->x2DataNum = dataCount;
    for (uint32_t i = 0; i < TRUNCATE_MOD_MAX_DIMS; ++i) {
        tilingData->outShape[i] = 1U;
        tilingData->x1Shape[i] = 1U;
        tilingData->x2Shape[i] = 1U;
    }
    tilingData->outShape[TRUNCATE_MOD_MAX_DIMS - 1U] = dataCount;
    tilingData->x1Shape[TRUNCATE_MOD_MAX_DIMS - 1U] = dataCount;
    tilingData->x2Shape[TRUNCATE_MOD_MAX_DIMS - 1U] = dataCount;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = truncate_mod<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./truncate_mod_data/int32_output_t_truncate_mod.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./truncate_mod_data/ && python3 compare_data.py 'int32'"), 0);
}
