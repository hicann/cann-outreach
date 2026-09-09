/*!
 * \file test_add.cpp
 * \brief Add 算子 kernel UT 测试
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "add_tiling.h"
#include "../../../op_kernel/add.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class AddKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AddKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "AddKernelTest TearDown" << endl;
    }
};

TEST_F(AddKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(AddTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    constexpr size_t zByteSize = 16384 * 4;
    std::vector<float> xHost(16384, 1);
    std::vector<float> yHost(16384, 1);
    std::vector<float> zHost(16384, 0);


    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(x, xHost.data(), xByteSize);
    memcpy(y, yHost.data(), yByteSize);

    // tiling 数据与 op_host/add_tiling.cpp 的赋值保持一致（totalLength、tileNum）
    AddTilingData* tilingData = reinterpret_cast<AddTilingData*>(tiling);
    tilingData->totalLength = size;
    tilingData->tileNum = 8;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 直接实例化 float 版本的模板核函数（与 tiling_key 的 DT_X=float 实例对应）
    ICPU_RUN_KF((add<float>), numBlocks, x, y, z, workspace, tiling);

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(zHost.data(), z, zByteSize);
    { std::ofstream _ofs("float32_output_add_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(zHost.data()), zByteSize); }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
