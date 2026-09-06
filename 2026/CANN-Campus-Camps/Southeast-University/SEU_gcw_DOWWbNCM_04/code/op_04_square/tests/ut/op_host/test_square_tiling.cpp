#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "square_tiling_data.h"

namespace SquareUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Square";

struct SquareTestParam {
    std::string caseName;
    std::initializer_list<int64_t> input_xShape;
    ge::DataType input_xDtype;
    ge::Format input_xFormat;
    std::initializer_list<int64_t> outputShape;
    ge::DataType outputDtype;
    ge::Format outputFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/square_tiling_key.h 和 op_host/square_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/square_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/square_tiling.cpp 中 GetWorkspaceSize 的逻辑
static SquareTestParam testCases[] = {
    {"square_0", {32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class SquareTilingTest : public testing::TestWithParam<SquareTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SquareTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SquareTilingTest TearDown." << std::endl;
    }
};

struct SquareCompileInfo {} compileInfo;

static void TestOneParamCase(const SquareTestParam &param)
{
    gert::StorageShape input_xShape = {param.input_xShape, param.input_xShape};
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{input_xShape, param.input_xDtype, param.input_xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outputShape, param.outputDtype, param.outputFormat}});
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

TEST_P(SquareTilingTest, tiling_test)
{
    const SquareTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SquareTilingTests,
    SquareTilingTest,
    testing::ValuesIn(testCases));

}
