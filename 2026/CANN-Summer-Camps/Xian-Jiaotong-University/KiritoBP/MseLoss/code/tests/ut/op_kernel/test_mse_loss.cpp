// test_mse_loss.cpp
// MseLoss 算子 Kernel UT (CPU 仿真)
// 参考: @wxy19931018 TruncateMod tests/ut/op_kernel/test_truncate_mod.cpp

#include "mse_loss_tiling.h"
#include "../../../op_kernel/mse_loss.cpp"

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

// ---- float→float16 转换 ----
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

// ---- Test Fixture ----
class MseLossKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "MseLossKernelTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "MseLossKernelTest TearDown" << endl;
    }
};

// ---- float32 测试 ----
TEST_F(MseLossKernelTest, test_kernel_run)
{
    constexpr size_t size = 8 * 2048;           // total elements
    constexpr size_t tilingDataSize = sizeof(MseLossTilingData);
    constexpr uint32_t numBlocks = 8;            // 模拟 8 核

    constexpr size_t dataByteSize = size * 4;    // float32

    // 构造 predict=1.0, label=2.0 → 期望 y=(1-2)^2=1.0
    vector<float> pHost(size, 1.0f);
    vector<float> lHost(size, 2.0f);
    vector<float> yHost(size, 0.0f);

    uint8_t* pDev = (uint8_t*)AscendC::GmAlloc(dataByteSize);
    uint8_t* lDev = (uint8_t*)AscendC::GmAlloc(dataByteSize);
    uint8_t* yDev = (uint8_t*)AscendC::GmAlloc(dataByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(pDev, pHost.data(), dataByteSize);
    memcpy(lDev, lHost.data(), dataByteSize);

    // 直接构造 tilingData
    MseLossTilingData* tilingData = reinterpret_cast<MseLossTilingData*>(tilingBuf);
    tilingData->length = size;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 运行 kernel (float32 实例, tiling key = 0)
    ICPU_RUN_KF((mse_loss<0>), numBlocks, pDev, lDev, yDev, workspace, tilingBuf);

    // 读回结果
    memcpy(yHost.data(), yDev, dataByteSize);

    // 写入 bin 文件供 compare_data.py 精度比对
    {
        ofstream ofs("float32_output_mse_loss_0.bin", ios::binary);
        ofs.write(reinterpret_cast<const char*>(yHost.data()), dataByteSize);
    }

    // GTest 断言: y 应全为 1.0
    for (size_t i = 0; i < size; i++) {
        ASSERT_NEAR(yHost[i], 1.0f, 1e-3f) << "Mismatch at index " << i;
    }

    AscendC::GmFree(pDev);
    AscendC::GmFree(lDev);
    AscendC::GmFree(yDev);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuf);
}
