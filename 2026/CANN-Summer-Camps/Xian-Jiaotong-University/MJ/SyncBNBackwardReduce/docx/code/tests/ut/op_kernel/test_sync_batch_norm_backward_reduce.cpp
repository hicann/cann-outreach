/*!
 * \file test_sync_batch_norm_backward_reduce.cpp
 * \brief SyncBatchNormBackwardReduce 算子 kernel UT 测试
 *
 * 参数化测试，覆盖3种dtype(float16/float32/bfloat16) × 多种shape × 单核/多核场景。
 * 从 gen_data.py 生成的 bin 文件读取输入数据，运行 kernel 后输出 bin 文件供 compare_data.py 比对。
 */

#include "sync_batch_norm_backward_reduce_tiling.h"
#include "../../../op_kernel/sync_batch_norm_backward_reduce.cpp"

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

// ============================================================
// 辅助函数：读写 bin 文件
// ============================================================

static bool ReadBinFile(const string& filename, uint8_t* buf, size_t size)
{
    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }
    ifs.read(reinterpret_cast<char*>(buf), size);
    if (static_cast<size_t>(ifs.gcount()) != size) {
        cerr << "ERROR: File size mismatch: " << filename
             << " (expected " << size << ", got " << ifs.gcount() << ")" << endl;
        return false;
    }
    return true;
}

static void WriteBinFile(const string& filename, const uint8_t* buf, size_t size)
{
    ofstream ofs(filename, ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf), size);
}

// ============================================================
// 测试参数结构体
// ============================================================

struct KernelTestParam {
    string caseName;       // 用例ID: "c00", "c01", ...
    string dtypeStr;       // "float16", "float32", "bfloat16"
    uint32_t tilingKey;    // 0=fp16, 1=fp32, 2=bf16
    int64_t totalNum;      // 总元素数量
    uint32_t numBlocks;    // CPU模拟核数
    int64_t blockFactor;   // 每核处理元素数
    int64_t ubFactor;      // 每次UB循环元素数
    size_t elemSize;       // 每元素字节数 (2或4)

    // 生成输入 bin 文件名
    string InputFile(const string& tensorName) const {
        return dtypeStr + "_" + caseName + "_input_sync_batch_norm_backward_reduce_" + tensorName + ".bin";
    }
    // 生成输出 bin 文件名
    string OutputFile(int outIdx) const {
        return dtypeStr + "_" + caseName + "_output_sync_batch_norm_backward_reduce_" + to_string(outIdx) + ".bin";
    }
};

// ============================================================
// 测试用例矩阵 (与 gen_data.py 对齐)
// blockFactor = CeilAlign(max(CeilDiv(totalNum, numBlocks), elemPerBlock), elemPerBlock)
// ubFactor: float16/bf16=3632, float32=4032
// 注意: CPU模拟环境UB限制为196608字节(192KB)，float32的ubFactor需≤4096
//       (48 bytes/elem × 4096 = 196608)。为留安全余量使用4032。
//       真实NPU的UB为262144，tiling代码会算出5456。
// ============================================================

static vector<KernelTestParam> kernelTestCases = {
    // float16 (tilingKey=0, elemSize=2, elemPerBlock=16, ubFactor=3632)
    {"c00", "float16",  0, 5,     1, 16,   3632, 2},  // 单核小shape
    {"c01", "float16",  0, 100,   1, 112,  3632, 2},  // 单核中shape
    {"c02", "float16",  0, 500,   1, 512,  3632, 2},  // 单核大shape
    {"c03", "float16",  0, 1024,  4, 256,  3632, 2},  // 多核阈值边界
    {"c04", "float16",  0, 4096,  8, 512,  3632, 2},  // 多核中shape
    {"c05", "float16",  0, 65536, 8, 8192, 3632, 2},  // 多核大shape
    // float32 (tilingKey=1, elemSize=4, elemPerBlock=8, ubFactor=4032)
    {"c06", "float32",  1, 5,     1, 8,    4032, 4},  // 单核小shape
    {"c07", "float32",  1, 4096,  8, 512,  4032, 4},  // 多核中shape
    {"c08", "float32",  1, 65536, 8, 8192, 4032, 4},  // 多核大shape
    // bfloat16 (tilingKey=2, elemSize=2, elemPerBlock=16, ubFactor=3632)
    {"c09", "bfloat16", 2, 5,     1, 16,   3632, 2},  // 单核小shape
    {"c10", "bfloat16", 2, 4096,  8, 512,  3632, 2},  // 多核中shape
    {"c11", "bfloat16", 2, 65536, 8, 8192, 3632, 2},  // 多核大shape
    // 多维shape (展平后同1D)
    {"c12", "float16",  0, 6,     1, 16,   3632, 2},  // {2,3} 单核
    {"c13", "float32",  1, 4096,  8, 512,  4032, 4},  // {4,1024} 多核
};

// ============================================================
// 参数化测试类
// ============================================================

class SyncBatchNormBackwardReduceKernelTest :
    public testing::TestWithParam<KernelTestParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "SyncBatchNormBackwardReduceKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SyncBatchNormBackwardReduceKernelTest TearDown" << endl;
    }
};

// ============================================================
// 参数化测试：运行 kernel 并输出 bin 文件
// ============================================================

TEST_P(SyncBatchNormBackwardReduceKernelTest, test_kernel_run)
{
    const auto& p = GetParam();
    const size_t totalBytes = static_cast<size_t>(p.totalNum) * p.elemSize;
    const size_t tilingDataSize = sizeof(SyncBatchNormBackwardReduceTilingData);

    cout << "\n--- Running case: " << p.caseName
         << " dtype=" << p.dtypeStr
         << " totalNum=" << p.totalNum
         << " numBlocks=" << p.numBlocks
         << " blockFactor=" << p.blockFactor
         << " ubFactor=" << p.ubFactor
         << " ---" << endl;

    // 分配 GM 内存
    uint8_t* sum_dy = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* sum_dy_dx_pad = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* invert_std = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* sum_dy_xmu = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(64);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ASSERT_NE(sum_dy, nullptr);
    ASSERT_NE(sum_dy_dx_pad, nullptr);
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(invert_std, nullptr);
    ASSERT_NE(sum_dy_xmu, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    // 从 bin 文件读取输入数据 (gen_data.py 生成)
    ASSERT_TRUE(ReadBinFile(p.InputFile("sum_dy"), sum_dy, totalBytes))
        << "Failed to read sum_dy input";
    ASSERT_TRUE(ReadBinFile(p.InputFile("sum_dy_dx_pad"), sum_dy_dx_pad, totalBytes))
        << "Failed to read sum_dy_dx_pad input";
    ASSERT_TRUE(ReadBinFile(p.InputFile("mean"), mean, totalBytes))
        << "Failed to read mean input";
    ASSERT_TRUE(ReadBinFile(p.InputFile("invert_std"), invert_std, totalBytes))
        << "Failed to read invert_std input";

    // 初始化输出缓冲区为 0
    memset(sum_dy_xmu, 0, totalBytes);
    memset(y, 0, totalBytes);

    // 构造 tiling 数据
    SyncBatchNormBackwardReduceTilingData* tilingData =
        reinterpret_cast<SyncBatchNormBackwardReduceTilingData*>(tiling);
    tilingData->totalNum = p.totalNum;
    tilingData->blockFactor = p.blockFactor;
    tilingData->ubFactor = p.ubFactor;

    // 设置 tiling key 和 kernel 模式
    ICPU_SET_TILING_KEY(p.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 根据 tilingKey 调用对应模板实例
    if (p.tilingKey == 0) {
        // float16
        ICPU_RUN_KF((sync_batch_norm_backward_reduce<0>), p.numBlocks,
                    sum_dy, sum_dy_dx_pad, mean, invert_std,
                    sum_dy_xmu, y, workspace, tiling);
    } else if (p.tilingKey == 1) {
        // float32
        ICPU_RUN_KF((sync_batch_norm_backward_reduce<1>), p.numBlocks,
                    sum_dy, sum_dy_dx_pad, mean, invert_std,
                    sum_dy_xmu, y, workspace, tiling);
    } else {
        // bfloat16
        ICPU_RUN_KF((sync_batch_norm_backward_reduce<2>), p.numBlocks,
                    sum_dy, sum_dy_dx_pad, mean, invert_std,
                    sum_dy_xmu, y, workspace, tiling);
    }

    // 保存输出到 bin 文件供 compare_data.py 比对
    WriteBinFile(p.OutputFile(0), sum_dy_xmu, totalBytes);  // sum_dy_xmu
    WriteBinFile(p.OutputFile(1), y, totalBytes);            // y

    cout << "--- Case " << p.caseName << " completed, output saved ---" << endl;

    // 释放资源
    AscendC::GmFree(sum_dy);
    AscendC::GmFree(sum_dy_dx_pad);
    AscendC::GmFree(mean);
    AscendC::GmFree(invert_std);
    AscendC::GmFree(sum_dy_xmu);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 注册参数化测试
INSTANTIATE_TEST_SUITE_P(
    SyncBatchNormBackwardReduceKernelTests,
    SyncBatchNormBackwardReduceKernelTest,
    testing::ValuesIn(kernelTestCases));
