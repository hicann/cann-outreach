/*!
 * \file test_truncate_mod.cpp
 * \brief TruncateMod 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "truncate_mod_tiling.h"
#include "../../../op_kernel/truncate_mod.cpp"

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

class TruncateModKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "TruncateModKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "TruncateModKernelTest TearDown" << endl;
    }
};

TEST_F(TruncateModKernelTest, test_kernel_run)
{
    constexpr size_t size = 5;
    constexpr size_t tilingDataSize = sizeof(TruncateModTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1ByteSize = 5 * 2;
    constexpr size_t x2ByteSize = 5 * 2;
    constexpr size_t yByteSize = 5 * 2;
    std::vector<float> x1Host(5, 1);
    std::vector<float> x2Host(5, 1);
    std::vector<float> yHost(5, 0);
    
    
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 5; _i++) { uint16_t _h = FloatToHalf(x1Host[_i]); memcpy(x1 + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 5; _i++) { uint16_t _h = FloatToHalf(x2Host[_i]); memcpy(x2 + _i * 2, &_h, 2); }
    
    // 直接构造 tilingData（固定值，生成时确定）
    TruncateModTilingData* tilingData = reinterpret_cast<TruncateModTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((truncate_mod<0>), numBlocks, x1, x2, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float16_output_truncate_mod_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
