#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "truncate_mod_tiling_data.h"

namespace TruncateModUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "TruncateMod";

struct TruncateModTestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
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

// TilingData 序列化格式：TruncateModTilingData 7 个 int64 依次为
//   totalNum blockFactor ubFactor x1InnerLen x1RowCount x2InnerLen x2RowCount
// 期望值按 truncate_mod_tiling.cpp 算法手工推导（UT 环境 coreNum=64, ubSize=262144）：
//   maxElemNum = (262144-1024)/24 = 10880（2B）或 /28 = 9325（4B）
//   ubFactor = FloorAlign(maxElemNum, 256/dtypeSize)，period 对齐时取 period 整数倍
static TruncateModTestParam testCases[] = {
    // T1: FP16 ELEWISE 既有用例回归（total=5, 每核 512 对齐）
    {"T1_fp16_elewise_small", {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {5}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL,
     "5 512 10880 5 1 5 1 ", {0}, 64, 262144, 4096},
    // T2: FP16 ELEWISE 多维（同 shape → innerLen=total, rowCount=1）
    {"T2_fp16_elewise_2d", {4, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 8}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {4, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL,
     "32 512 10880 32 1 32 1 ", {0}, 64, 262144, 4096},
    // T3: BF16 ELEWISE（period=1024 触发对齐优化：ubFactor=10240）
    {"T3_bf16_elewise", {1024}, ge::DT_BF16, ge::FORMAT_ND, {1024}, ge::DT_BF16, ge::FORMAT_ND,
     {1024}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL,
     "1024 1024 10240 1024 1 1024 1 ", {0}, 64, 262144, 4096},
    // T4: FP32 ELEWISE（period=1024 触发对齐优化：ubFactor=9216）
    {"T4_fp32_elewise", {1024}, ge::DT_FLOAT, ge::FORMAT_ND, {1024}, ge::DT_FLOAT, ge::FORMAT_ND,
     {1024}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL,
     "1024 1024 9216 1024 1 1024 1 ", {0}, 64, 262144, 4096},
    // T5: FP32 标量广播（x2=[1] → innerLen=1, rowCount=1；minPeriod=1 不触发对齐优化）
    {"T5_fp32_scalar_bcast", {1024}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND,
     {1024}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL,
     "1024 1024 9280 1024 1 1 1 ", {0}, 64, 262144, 4096},
    // T6: FP16 行广播（x1=[1,64] 行重复 → 单行 tile：ubFactor=gcd(64)=64；输出 [64,64]）
    {"T6_fp16_row_bcast", {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL,
     "4096 2048 64 64 1 4096 1 ", {0}, 64, 262144, 4096},
    // T7: FP32 左侧前缀广播（x1=[1,1,32] 行重复 → 单行 tile：ubFactor=gcd(32)=32；输出 [4,5,32]）
    {"T7_fp32_left_prefix_bcast", {1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 5, 32}, ge::DT_FLOAT, ge::FORMAT_ND,
     {4, 5, 32}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL,
     "640 1024 32 32 1 640 1 ", {0}, 64, 262144, 4096},
    // T8: BF16 全标量（rank-0 → EnsureNotScalar 语义）
    {"T8_bf16_scalar_scalar", {}, ge::DT_BF16, ge::FORMAT_ND, {}, ge::DT_BF16, ge::FORMAT_ND,
     {}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL,
     "1 512 10880 1 1 1 1 ", {0}, 64, 262144, 4096},
    // T9: 非法广播（[2,3]×[4] → dim 冲突，tiling 防御性校验拒绝）
    {"T9_invalid_bcast", {2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_FAILED, 0UL,
     "", {0}, 64, 262144, 4096},
    // T10: FP32 大 shape 多核切分（coreNumByData=977 → usedCore=64；blockFactor=15872）
    {"T10_fp32_large", {1000000}, ge::DT_FLOAT, ge::FORMAT_ND, {1000000}, ge::DT_FLOAT, ge::FORMAT_ND,
     {1000000}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL,
     "1000000 15872 9280 1000000 1 1000000 1 ", {0}, 64, 262144, 4096},
    // T11: FP32 period 对齐优化（period=8000, 8000%64==0 → ubFactor=8000）
    {"T11_fp32_period_align", {1000, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {1000, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
     {1000, 8}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL,
     "8000 1024 8000 8000 1 8000 1 ", {0}, 64, 262144, 4096},
    // T12: xfail 列广播（x1=[64,1] rowCount=64>1 → row-run 模型拒绝，PLAN.md §5 design_issue）
    {"T12_xfail_column_bcast", {64, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT, ge::FORMAT_ND,
     {64, 64}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_FAILED, 0UL,
     "", {0}, 64, 262144, 4096},
};

class TruncateModTilingTest : public testing::TestWithParam<TruncateModTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "TruncateModTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "TruncateModTilingTest TearDown." << std::endl;
    }
};

struct TruncateModCompileInfo {} compileInfo;

static void TestOneParamCase(const TruncateModTestParam &param)
{
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
        {x2Shape, param.x2Dtype, param.x2Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
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

TEST_P(TruncateModTilingTest, tiling_test)
{
    const TruncateModTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    TruncateModTilingTests,
    TruncateModTilingTest,
    testing::ValuesIn(testCases));

}
