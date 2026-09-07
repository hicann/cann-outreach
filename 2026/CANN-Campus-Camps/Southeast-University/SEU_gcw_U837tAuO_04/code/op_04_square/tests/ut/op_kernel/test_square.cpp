/*!
 * \file test_square.cpp
 * \brief Square 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "square_tiling.h"
#include "../../../op_kernel/square.cpp"

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

class SquareKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SquareKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SquareKernelTest TearDown" << endl;
    }
};

TEST_F(SquareKernelTest, test_kernel_run)
{
    constexpr size_t size = 32;
    constexpr size_t tilingDataSize = sizeof(SquareTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t input_xByteSize = 32 * 2;
    constexpr size_t outputByteSize = 32 * 2;
    std::vector<float> input_xHost(32, 1);
    std::vector<float> outputHost(32, 0);
    
    
    uint8_t* input_x = (uint8_t*)AscendC::GmAlloc(input_xByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 32; _i++) { uint16_t _h = FloatToHalf(input_xHost[_i]); memcpy(input_x + _i * 2, &_h, 2); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/square_tiling_data.h 中的字段定义
    //         参考 op_host/square_tiling.cpp 中的 tiling 计算逻辑
    SquareTilingData* tilingData = reinterpret_cast<SquareTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/square_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((square<0>), numBlocks, input_x, output, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outputHost.data(), output, outputByteSize);
    { std::ofstream _ofs("float16_output_square_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outputHost.data()), outputByteSize); }
    
    AscendC::GmFree(input_x);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
