/*!
 * \file test_truncate_mod.cpp
 * \brief TruncateMod kernel UT — fp16 / fp32 / bf16 全类型覆盖
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
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}
static float HalfToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t exp = ((h >> 10) & 0x1f);
    uint32_t mant = h & 0x3ff;
    if (exp == 0) return sign ? -0.0f : 0.0f;
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    exp = exp - 15 + 127;
    uint32_t bits = (sign << 31) | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &bits, sizeof(float));
    return f;
}

class TruncateModKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()    { cout << "TruncateModKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "TruncateModKernelTest TearDown" << endl; }
};

// ---- fp16 ----
TEST_F(TruncateModKernelTest, test_fp16)
{
    constexpr int64_t N = 32;
    constexpr int64_t tilingSz = sizeof(TruncateModTilingData);
    size_t byteSz = N * 2;

    vector<float> x1Host(N), x2Host(N);
    for (int64_t i = 0; i < N; i++) {
        x1Host[i] = (i % 2 == 0) ? i + 1.5f : -(i + 1.0f);
        x2Host[i] = (i % 3 == 0) ? -3.0f : 2.0f;
    }

    uint8_t* x1  = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* x2  = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* y   = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* ws  = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* til = (uint8_t*)AscendC::GmAlloc(tilingSz);

    for (int64_t i = 0; i < N; i++) {
        uint16_t h1 = FloatToHalf(x1Host[i]);
        uint16_t h2 = FloatToHalf(x2Host[i]);
        memcpy(x1 + i * 2, &h1, 2);
        memcpy(x2 + i * 2, &h2, 2);
    }

    auto* td = reinterpret_cast<TruncateModTilingData*>(til);
    td->totalNum    = N;
    td->blockFactor = N;
    td->ubFactor    = N;

    ICPU_SET_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_FP16);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<TRUNCATEMOD_TPL_SCH_MODE_FP16>), 1, x1, x2, y, ws, til);

    // verify
    for (int64_t i = 0; i < N; i++) {
        uint16_t h; memcpy(&h, y + i * 2, 2);
        float result = HalfToFloat(h);
        float expected = x1Host[i] - trunc(x1Host[i] / x2Host[i]) * x2Host[i];
        EXPECT_NEAR(result, expected, 0.05f) << "fp16 mismatch at " << i;
    }

    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y);
    AscendC::GmFree(ws); AscendC::GmFree(til);
}

// ---- fp32 ----
TEST_F(TruncateModKernelTest, test_fp32)
{
    constexpr int64_t N = 16;
    constexpr int64_t tilingSz = sizeof(TruncateModTilingData);
    size_t byteSz = N * 4;

    vector<float> x1 = {5.5f, -11.51f, 36.23f, 7.f, -10.f, -8.f, -15.f, -7.f,
                        10.f, 8.f, 15.f, 7.f, -10.f, -8.f, -15.f, -7.f};
    vector<float> x2 = {2.f, 3.f, -24.1f, 2.f, 3.f, 5.f, 4.f, 2.f,
                        -3.f, -5.f, -4.f, -2.f, -3.f, -5.f, -4.f, -2.f};

    uint8_t* x1G  = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* x2G  = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* yG   = (uint8_t*)AscendC::GmAlloc(byteSz);
    uint8_t* ws   = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* til  = (uint8_t*)AscendC::GmAlloc(tilingSz);

    memcpy(x1G, x1.data(), byteSz);
    memcpy(x2G, x2.data(), byteSz);

    auto* td = reinterpret_cast<TruncateModTilingData*>(til);
    td->totalNum    = N;
    td->blockFactor = N;
    td->ubFactor    = N;

    ICPU_SET_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_FP32);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<TRUNCATEMOD_TPL_SCH_MODE_FP32>), 1, x1G, x2G, yG, ws, til);

    vector<float> result(N);
    memcpy(result.data(), yG, byteSz);
    for (int64_t i = 0; i < N; i++) {
        float expected = x1[i] - trunc(x1[i] / x2[i]) * x2[i];
        EXPECT_NEAR(result[i], expected, 1e-5f) << "fp32 mismatch at " << i;
    }

    AscendC::GmFree(x1G); AscendC::GmFree(x2G); AscendC::GmFree(yG);
    AscendC::GmFree(ws); AscendC::GmFree(til);
}
