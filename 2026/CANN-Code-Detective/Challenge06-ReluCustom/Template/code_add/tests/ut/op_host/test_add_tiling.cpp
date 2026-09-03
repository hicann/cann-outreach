#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "add_tiling_data.h"

namespace AddUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Add";

struct AddTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> zShape;
    ge::DataType zDtype;
    ge::Format zFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// 期望值对应 op_host/add_tiling.cpp 的简化实现：
//   tilingData = {totalLength=16384, tileNum=8}，workspace = {0}
//   expectTilingKey：DT_X=float 对应 tiling_key 中 SEL 的第 0 个实例
//   （若与框架实际分配的 key 不一致，以 op_host/add_tiling.cpp 实际运行值为准）
static AddTestParam testCases[] = {
    {"add_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "16384 8 ", {0}, 64, 262144, 4096},
};

class AddTilingTest : public testing::TestWithParam<AddTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "AddTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "AddTilingTest TearDown." << std::endl;
    }
};

struct AddCompileInfo {} compileInfo;

static void TestOneParamCase(const AddTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape zShape = {param.zShape, param.zShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}, {yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{zShape, param.zDtype, param.zFormat}});
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

TEST_P(AddTilingTest, tiling_test)
{
    const AddTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    AddTilingTests,
    AddTilingTest,
    testing::ValuesIn(testCases));

}
