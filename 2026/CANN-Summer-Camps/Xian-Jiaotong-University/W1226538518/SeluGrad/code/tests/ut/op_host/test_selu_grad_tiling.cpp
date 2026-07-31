#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "selu_grad_tiling_data.h"

namespace SeluGradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "SeluGrad";

struct SeluGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradientsShape;
    ge::DataType gradientsDtype;
    ge::Format gradientsFormat;
    std::initializer_list<int64_t> outputsShape;
    ge::DataType outputsDtype;
    ge::Format outputsFormat;
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

static SeluGradTestParam testCases[] = {
    {"selu_grad_0", {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class SeluGradTilingTest : public testing::TestWithParam<SeluGradTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SeluGradTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SeluGradTilingTest TearDown." << std::endl;
    }
};

struct SeluGradCompileInfo {} compileInfo;

static void TestOneParamCase(const SeluGradTestParam &param)
{
    gert::StorageShape gradientsShape = {param.gradientsShape, param.gradientsShape};
    gert::StorageShape outputsShape = {param.outputsShape, param.outputsShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{gradientsShape, param.gradientsDtype, param.gradientsFormat},
        {outputsShape, param.outputsDtype, param.outputsFormat}});
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

TEST_P(SeluGradTilingTest, tiling_test)
{
    const SeluGradTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SeluGradTilingTests,
    SeluGradTilingTest,
    testing::ValuesIn(testCases));

}
