/*!
 * \file test_truncate_mod.cpp
 * \brief TruncateMod kernel UT — 6 dtypes, reads from gen_data bins
 */
#include "truncate_mod_tiling.h"
#include "../../../op_kernel/truncate_mod.cpp"

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;
namespace fs = std::filesystem;

static vector<uint8_t> readBin(const string& path) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

class TruncateModKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "[UT] TruncateModKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "[UT] TruncateModKernelTest TearDown" << endl; }
};

// ====== float16 (half, key=2) ======
TEST_F(TruncateModKernelTest, dtype_float16) {
    string dtype = "float16";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen / 2;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(2);  // (C_DT_FLOAT16, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<half, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}

// ====== float32 (float, key=0) ======
TEST_F(TruncateModKernelTest, dtype_float32) {
    string dtype = "float32";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen / 4;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(0);  // (C_DT_FLOAT, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<float, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}

// ====== bfloat16 (bfloat16_t, key=4) ======
TEST_F(TruncateModKernelTest, dtype_bf16) {
    string dtype = "bfloat16";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen / 2;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(4);  // (C_DT_BF16, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<bfloat16_t, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}

// ====== int32 (int32_t, key=6) ======
TEST_F(TruncateModKernelTest, dtype_int32) {
    string dtype = "int32";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen / 4;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(6);  // (C_DT_INT32, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<int32_t, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}

// ====== int8 (int8_t, key=8) ======
TEST_F(TruncateModKernelTest, dtype_int8) {
    string dtype = "int8";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(8);  // (C_DT_INT8, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<int8_t, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}

// ====== uint8 (uint8_t, key=10) ======
TEST_F(TruncateModKernelTest, dtype_uint8) {
    string dtype = "uint8";
    string dataDir = fs::current_path().string() + "/../../op_kernel/truncate_mod_data";
    auto x1b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x1.bin");
    auto x2b = readBin(dataDir + "/" + dtype + "_input_truncate_mod_x2.bin");
    ASSERT_FALSE(x1b.empty()) << "Missing input bin for " << dtype;
    size_t blen = x1b.size(), ec = blen;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* y  = (uint8_t*)AscendC::GmAlloc(blen);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));
    memcpy(x1, x1b.data(), blen); memcpy(x2, x2b.data(), blen);

    auto* td = reinterpret_cast<TruncateModTilingData*>(tl);
    memset(td, 0, sizeof(TruncateModTilingData));
    td->totalNum = ec; td->blockFactor = ec; td->ubFactor = ec;
    td->outDims=1; td->x1Dims=1; td->x2Dims=1;
    td->outShape[0]=ec; td->x1Shape[0]=ec; td->x2Shape[0]=ec;

    ICPU_SET_TILING_KEY(10);  // (C_DT_UINT8, DBUF_MODE=0)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((truncate_mod<uint8_t, 0>), 1u, x1, x2, y, ws, tl);

    ofstream(dataDir + "/" + dtype + "_output_truncate_mod_0.bin", ios::binary)
        .write(reinterpret_cast<const char*>(y), blen);
    cout << "[PASS] " << dtype << " elemCount=" << ec << endl;
    AscendC::GmFree(x1); AscendC::GmFree(x2); AscendC::GmFree(y); AscendC::GmFree(ws); AscendC::GmFree(tl);
}