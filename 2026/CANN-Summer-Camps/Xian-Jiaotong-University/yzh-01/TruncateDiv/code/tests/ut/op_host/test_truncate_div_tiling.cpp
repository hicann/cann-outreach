#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "truncate_div_tiling_data.h"

namespace TruncateDivUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "TruncateDiv";

struct TruncateDivTestParam {
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

static TruncateDivTestParam testCases[] = {
    {"truncate_div_0", {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class TruncateDivTilingTest : public testing::TestWithParam<TruncateDivTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "TruncateDivTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "TruncateDivTilingTest TearDown." << std::endl;
    }
};

struct TruncateDivCompileInfo {} compileInfo;

static void TestOneParamCase(const TruncateDivTestParam &param)
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

TEST_P(TruncateDivTilingTest, tiling_test)
{
    const TruncateDivTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    TruncateDivTilingTests,
    TruncateDivTilingTest,
    testing::ValuesIn(testCases));

}
