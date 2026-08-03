#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "sync_batch_norm_backward_reduce_tiling_data.h"

namespace SyncBatchNormBackwardReduceUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "SyncBatchNormBackwardReduce";

struct SyncBatchNormBackwardReduceTestParam {
    std::string caseName;
    std::initializer_list<int64_t> sum_dyShape;
    ge::DataType sum_dyDtype;
    ge::Format sum_dyFormat;
    std::initializer_list<int64_t> sum_dy_dx_padShape;
    ge::DataType sum_dy_dx_padDtype;
    ge::Format sum_dy_dx_padFormat;
    std::initializer_list<int64_t> meanShape;
    ge::DataType meanDtype;
    ge::Format meanFormat;
    std::initializer_list<int64_t> invert_stdShape;
    ge::DataType invert_stdDtype;
    ge::Format invert_stdFormat;
    std::initializer_list<int64_t> sum_dy_xmuShape;
    ge::DataType sum_dy_xmuDtype;
    ge::Format sum_dy_xmuFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// ============================================================
// 测试用例矩阵 (16个用例)
// 覆盖: 3种dtype × 单核(totalNum<1024)/多核(>=1024) × 1D/多维 shape
//
// Tiling 计算公式 (ubSize=262144, coreNum=64):
//   elemPerBlock = 32 / typeLen   (fp16/bf16=16, fp32=8)
//   bytesPerElem = (FLOAT) ? 48 : 72
//   ubFactor = (ubSize / bytesPerElem / elemPerBlock) * elemPerBlock
//     fp16/bf16: (262144/72/16)*16 = 3632
//     fp32:      (262144/48/8)*8   = 5456
//   totalNum < 1024 → useCoreNum = 1
//   totalNum >= 1024 → useCoreNum = min(64, CeilDiv(totalNum, elemPerBlock))
//   blockFactor = CeilAlign(max(CeilDiv(totalNum, useCoreNum), elemPerBlock), elemPerBlock)
//
// TilingKey: float16=0, float32=1, bfloat16=2
// ============================================================

static SyncBatchNormBackwardReduceTestParam testCases[] = {
    // === float16 (TilingKey=0, elemPerBlock=16, ubFactor=3632) ===
    // 单核 (totalNum < 1024)
    {"fp16_s5",      {5},     DT_FLOAT16, FORMAT_ND, {5},     DT_FLOAT16, FORMAT_ND, {5},     DT_FLOAT16, FORMAT_ND, {5},     DT_FLOAT16, FORMAT_ND, {5},     DT_FLOAT16, FORMAT_ND, {5},     DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "5 16 3632 ",       {0}, 64, 262144, 4096},
    {"fp16_s100",    {100},   DT_FLOAT16, FORMAT_ND, {100},   DT_FLOAT16, FORMAT_ND, {100},   DT_FLOAT16, FORMAT_ND, {100},   DT_FLOAT16, FORMAT_ND, {100},   DT_FLOAT16, FORMAT_ND, {100},   DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "100 112 3632 ",    {0}, 64, 262144, 4096},
    {"fp16_s500",    {500},   DT_FLOAT16, FORMAT_ND, {500},   DT_FLOAT16, FORMAT_ND, {500},   DT_FLOAT16, FORMAT_ND, {500},   DT_FLOAT16, FORMAT_ND, {500},   DT_FLOAT16, FORMAT_ND, {500},   DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "500 512 3632 ",    {0}, 64, 262144, 4096},
    // 多核 (totalNum >= 1024)
    {"fp16_s1024",   {1024},  DT_FLOAT16, FORMAT_ND, {1024},  DT_FLOAT16, FORMAT_ND, {1024},  DT_FLOAT16, FORMAT_ND, {1024},  DT_FLOAT16, FORMAT_ND, {1024},  DT_FLOAT16, FORMAT_ND, {1024},  DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "1024 16 3632 ",    {0}, 64, 262144, 4096},
    {"fp16_s4096",   {4096},  DT_FLOAT16, FORMAT_ND, {4096},  DT_FLOAT16, FORMAT_ND, {4096},  DT_FLOAT16, FORMAT_ND, {4096},  DT_FLOAT16, FORMAT_ND, {4096},  DT_FLOAT16, FORMAT_ND, {4096},  DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "4096 64 3632 ",    {0}, 64, 262144, 4096},
    {"fp16_s65536",  {65536}, DT_FLOAT16, FORMAT_ND, {65536}, DT_FLOAT16, FORMAT_ND, {65536}, DT_FLOAT16, FORMAT_ND, {65536}, DT_FLOAT16, FORMAT_ND, {65536}, DT_FLOAT16, FORMAT_ND, {65536}, DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "65536 1024 3632 ", {0}, 64, 262144, 4096},
    // 多维shape
    {"fp16_s2x3",    {2, 3},  DT_FLOAT16, FORMAT_ND, {2, 3},  DT_FLOAT16, FORMAT_ND, {2, 3},  DT_FLOAT16, FORMAT_ND, {2, 3},  DT_FLOAT16, FORMAT_ND, {2, 3},  DT_FLOAT16, FORMAT_ND, {2, 3},  DT_FLOAT16, FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 0UL, "6 16 3632 ",       {0}, 64, 262144, 4096},

    // === float32 (TilingKey=1, elemPerBlock=8, ubFactor=5456) ===
    // 单核
    {"fp32_s5",      {5},     DT_FLOAT,   FORMAT_ND, {5},     DT_FLOAT,   FORMAT_ND, {5},     DT_FLOAT,   FORMAT_ND, {5},     DT_FLOAT,   FORMAT_ND, {5},     DT_FLOAT,   FORMAT_ND, {5},     DT_FLOAT,   FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 1UL, "5 8 5456 ",        {0}, 64, 262144, 4096},
    {"fp32_s500",    {500},   DT_FLOAT,   FORMAT_ND, {500},   DT_FLOAT,   FORMAT_ND, {500},   DT_FLOAT,   FORMAT_ND, {500},   DT_FLOAT,   FORMAT_ND, {500},   DT_FLOAT,   FORMAT_ND, {500},   DT_FLOAT,   FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 1UL, "500 504 5456 ",    {0}, 64, 262144, 4096},
    // 多核
    {"fp32_s1024",   {1024},  DT_FLOAT,   FORMAT_ND, {1024},  DT_FLOAT,   FORMAT_ND, {1024},  DT_FLOAT,   FORMAT_ND, {1024},  DT_FLOAT,   FORMAT_ND, {1024},  DT_FLOAT,   FORMAT_ND, {1024},  DT_FLOAT,   FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 1UL, "1024 16 5456 ",    {0}, 64, 262144, 4096},
    {"fp32_s4096",   {4096},  DT_FLOAT,   FORMAT_ND, {4096},  DT_FLOAT,   FORMAT_ND, {4096},  DT_FLOAT,   FORMAT_ND, {4096},  DT_FLOAT,   FORMAT_ND, {4096},  DT_FLOAT,   FORMAT_ND, {4096},  DT_FLOAT,   FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 1UL, "4096 64 5456 ",    {0}, 64, 262144, 4096},
    {"fp32_s65536",  {65536}, DT_FLOAT,   FORMAT_ND, {65536}, DT_FLOAT,   FORMAT_ND, {65536}, DT_FLOAT,   FORMAT_ND, {65536}, DT_FLOAT,   FORMAT_ND, {65536}, DT_FLOAT,   FORMAT_ND, {65536}, DT_FLOAT,   FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 1UL, "65536 1024 5456 ", {0}, 64, 262144, 4096},

    // === bfloat16 (TilingKey=2, elemPerBlock=16, ubFactor=3632) ===
    // 单核
    {"bf16_s5",      {5},     DT_BF16,    FORMAT_ND, {5},     DT_BF16,    FORMAT_ND, {5},     DT_BF16,    FORMAT_ND, {5},     DT_BF16,    FORMAT_ND, {5},     DT_BF16,    FORMAT_ND, {5},     DT_BF16,    FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 2UL, "5 16 3632 ",       {0}, 64, 262144, 4096},
    {"bf16_s500",    {500},   DT_BF16,    FORMAT_ND, {500},   DT_BF16,    FORMAT_ND, {500},   DT_BF16,    FORMAT_ND, {500},   DT_BF16,    FORMAT_ND, {500},   DT_BF16,    FORMAT_ND, {500},   DT_BF16,    FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 2UL, "500 512 3632 ",    {0}, 64, 262144, 4096},
    // 多核
    {"bf16_s4096",   {4096},  DT_BF16,    FORMAT_ND, {4096},  DT_BF16,    FORMAT_ND, {4096},  DT_BF16,    FORMAT_ND, {4096},  DT_BF16,    FORMAT_ND, {4096},  DT_BF16,    FORMAT_ND, {4096},  DT_BF16,    FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 2UL, "4096 64 3632 ",    {0}, 64, 262144, 4096},
    {"bf16_s65536",  {65536}, DT_BF16,    FORMAT_ND, {65536}, DT_BF16,    FORMAT_ND, {65536}, DT_BF16,    FORMAT_ND, {65536}, DT_BF16,    FORMAT_ND, {65536}, DT_BF16,    FORMAT_ND, {65536}, DT_BF16,    FORMAT_ND, "Ascend910B", GRAPH_SUCCESS, 2UL, "65536 1024 3632 ", {0}, 64, 262144, 4096},
};

class SyncBatchNormBackwardReduceTilingTest : public testing::TestWithParam<SyncBatchNormBackwardReduceTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SyncBatchNormBackwardReduceTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SyncBatchNormBackwardReduceTilingTest TearDown." << std::endl;
    }
};

struct SyncBatchNormBackwardReduceCompileInfo {} compileInfo;

static void TestOneParamCase(const SyncBatchNormBackwardReduceTestParam &param)
{
    gert::StorageShape sum_dyShape = {param.sum_dyShape, param.sum_dyShape};
    gert::StorageShape sum_dy_dx_padShape = {param.sum_dy_dx_padShape, param.sum_dy_dx_padShape};
    gert::StorageShape meanShape = {param.meanShape, param.meanShape};
    gert::StorageShape invert_stdShape = {param.invert_stdShape, param.invert_stdShape};
    gert::StorageShape sum_dy_xmuShape = {param.sum_dy_xmuShape, param.sum_dy_xmuShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{sum_dyShape, param.sum_dyDtype, param.sum_dyFormat},
        {sum_dy_dx_padShape, param.sum_dy_dx_padDtype, param.sum_dy_dx_padFormat},
        {meanShape, param.meanDtype, param.meanFormat},
        {invert_stdShape, param.invert_stdDtype, param.invert_stdFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{sum_dy_xmuShape, param.sum_dy_xmuDtype, param.sum_dy_xmuFormat},
        {yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(SyncBatchNormBackwardReduceTilingTest, tiling_test)
{
    const SyncBatchNormBackwardReduceTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SyncBatchNormBackwardReduceTilingTests,
    SyncBatchNormBackwardReduceTilingTest,
    testing::ValuesIn(testCases));

}
