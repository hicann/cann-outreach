/*!
 * \file test_truncate_mod.cpp
 * \brief TruncateMod 算子 kernel UT 测试（数据驱动，K1-K10）
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT。
 * 输入/输出 bin 文件由 gen_data.py 生成、compare_data.py 比对；
 * 数据目录通过 argv[1] 传入（run.sh 约定）。
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
#include <string>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

// 用例参数：与 gen_data.py 中 case 的 dtype/形状/字节数一一对应
struct KernelCase {
    const char* caseId;      // bin 文件前缀（如 "K1"）
    uint32_t tilingKey;      // 0=FP16, 1=BF16, 2=FP32
    int64_t totalNum;        // 输出总元素数
    int64_t blockFactor;     // 每核元素数
    int64_t ubFactor;        // 每次 UB 循环元素数
    int64_t x1InnerLen;      // x1 连续运行长度
    int64_t x1RowCount;
    int64_t x2InnerLen;
    int64_t x2RowCount;
    size_t dtypeSize;        // 输入/输出元素字节数
    uint32_t numBlocks;      // 核数
};

// tiling 数值与 op_host/truncate_mod_tiling.cpp 算法一致（coreNum=64, ubSize=196608=192KB）：
//   maxElemNum = (196608-1024)/24 = 8149（2B）或 /28 = 6985（4B）
//   fp16/bf16 ubFactor = floor_align(8149, 128) = 8064；fp32 ubFactor = floor_align(6985, 64) = 6976
//   （DESIGN.md §1.5 数值示例：mode0/1≈8064、mode2≈6976）
static const KernelCase kernelCases[] = {
    // K1: FP16 ELEWISE 非 32B 对齐（total=5, 单块 512 对齐）
    {"float16_K1", 0, 5, 512, 8064, 5, 1, 5, 1, 2, 1},
    // K2: FP16 ELEWISE 多核（blockFactor=1024 → 8 块）
    {"float16_K2", 0, 8192, 1024, 8064, 8192, 1, 8192, 1, 2, 8},
    // K3: BF16 ELEWISE 多核（period=8192 无对齐优化：8192 > maxElemNum=8149）
    {"bfloat16_K3", 1, 8192, 1024, 8064, 8192, 1, 8192, 1, 2, 8},
    // K4: FP32 ELEWISE 多核（period=8192 无对齐优化：8192 > maxElemNum=6985）
    {"float32_K4", 2, 8192, 1024, 6976, 8192, 1, 8192, 1, 4, 8},
    // K5: FP32 行广播（x1=[1,64] 行重复 → 单行 tile：ubFactor=64；x2 ELEWISE）
    {"float32_K5", 2, 4096, 2048, 64, 64, 1, 4096, 1, 4, 2},
    // K6: FP16 标量广播（x2 标量 → 策略 0）
    {"float16_K6", 0, 1024, 1024, 8064, 1024, 1, 1, 1, 2, 1},
    // K7: FP32 正负混合 + x2=0 边界（IEEE NaN）
    {"float32_K7", 2, 2048, 1024, 6976, 2048, 1, 2048, 1, 4, 2},
    // K8: FP32 大商值（|q|>=2^31，FP32 全链）
    {"float32_K8", 2, 1024, 1024, 6976, 1024, 1, 1024, 1, 4, 1},
    // K9: 非对齐 tile + 尾块（total=1003，三 dtype）
    {"float16_K9_fp16", 0, 1003, 1024, 8064, 1003, 1, 1003, 1, 2, 1},
    {"bfloat16_K9_bf16", 1, 1003, 1024, 8064, 1003, 1, 1003, 1, 2, 1},
    {"float32_K9_fp32", 2, 1003, 1024, 6976, 1003, 1, 1003, 1, 4, 1},
    // K10: BF16 左侧前缀广播（x1=[1,1,32] 行重复 → 单行 tile：ubFactor=32；x2 ELEWISE）
    {"bfloat16_K10", 1, 640, 1024, 32, 32, 1, 640, 1, 2, 1},
};

static std::string g_dataDir; // argv[1]

static bool ReadBin(const std::string& name, std::vector<uint8_t>& buf)
{
    std::string path = g_dataDir + "/" + name;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "ERROR: cannot open " << path << std::endl;
        return false;
    }
    buf.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return true;
}

static bool WriteBin(const std::string& name, const void* data, size_t bytes)
{
    std::string path = g_dataDir + "/" + name;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "ERROR: cannot write " << path << std::endl;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return true;
}

static void RunOneCase(const KernelCase& c)
{
    const std::string prefix = std::string(c.caseId);
    const size_t x1Bytes = c.totalNum * c.dtypeSize; // 分配与 buffer 声明等长（kernel 只访问 < innerLen）
    const size_t x2Bytes = c.totalNum * c.dtypeSize;
    const size_t yBytes = c.totalNum * c.dtypeSize;
    const size_t tilingBytes = sizeof(TruncateModTilingData);

    std::vector<uint8_t> x1Buf, x2Buf;
    ASSERT_TRUE(ReadBin(prefix + "_input_x1.bin", x1Buf));
    ASSERT_TRUE(ReadBin(prefix + "_input_x2.bin", x2Buf));

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1Bytes);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2Bytes);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingBytes);

    memcpy(x1, x1Buf.data(), x1Buf.size());
    memcpy(x2, x2Buf.data(), x2Buf.size());
    memset(y, 0, yBytes);

    TruncateModTilingData* tilingData = reinterpret_cast<TruncateModTilingData*>(tiling);
    tilingData->totalNum = c.totalNum;
    tilingData->blockFactor = c.blockFactor;
    tilingData->ubFactor = c.ubFactor;
    tilingData->x1InnerLen = c.x1InnerLen;
    tilingData->x1RowCount = c.x1RowCount;
    tilingData->x2InnerLen = c.x2InnerLen;
    tilingData->x2RowCount = c.x2RowCount;

    ICPU_SET_TILING_KEY(c.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    switch (c.tilingKey) {
        case 0:
            ICPU_RUN_KF((truncate_mod<0>), c.numBlocks, x1, x2, y, workspace, tiling);
            break;
        case 1:
            ICPU_RUN_KF((truncate_mod<1>), c.numBlocks, x1, x2, y, workspace, tiling);
            break;
        case 2:
            ICPU_RUN_KF((truncate_mod<2>), c.numBlocks, x1, x2, y, workspace, tiling);
            break;
        default:
            FAIL() << "bad tilingKey " << c.tilingKey;
    }

    ASSERT_TRUE(WriteBin(prefix + "_output.bin", y, yBytes));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

class TruncateModKernelTest : public testing::TestWithParam<KernelCase> {
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

TEST_P(TruncateModKernelTest, kernel_run)
{
    RunOneCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TruncateModKernelTests, TruncateModKernelTest,
                         testing::ValuesIn(kernelCases));

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1) {
        g_dataDir = argv[1];
    } else {
        g_dataDir = ".";
    }
    return RUN_ALL_TESTS();
}
