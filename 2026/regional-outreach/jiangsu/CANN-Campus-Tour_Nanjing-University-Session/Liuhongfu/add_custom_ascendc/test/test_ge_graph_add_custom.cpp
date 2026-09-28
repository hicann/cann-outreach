/**
 * @file test_ge_graph_add_custom.cpp
 * @brief add_custom 算子的 GE 图模式（单算子图）调用样例。
 *
 * 依赖：
 *   1) 算子工程已编译并安装（./build.sh --install），自定义算子包已设置 ASCEND_CUSTOM_OPP_PATH；
 *   2) 编译时需要能 include 到算子工程生成的算子原型头文件 add_custom.h
 *      （位于算子工程 build_out 或自定义算子包的 op_proto/inc 目录下）。
 *      头文件里为 AddCustom 生成了 set_input_x / set_input_y / update_output_desc_z 等接口。
 *
 * 运行：./test_ge_graph_add_custom [N2] [N1]
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

// GE 算子原型头文件（由算子工程编译生成）
#include "add_custom.h"

#include "utils.h"

int main(int argc, char** argv)
{
    int64_t n2 = 8;
    int64_t n1 = 257;
    if (argc > 1) { n2 = std::atoll(argv[1]); }
    if (argc > 2) { n1 = std::atoll(argv[2]); }

    const int64_t elementCount = n2 * n1;
    const size_t byteSize = static_cast<size_t>(elementCount) * sizeof(uint16_t);
    std::printf("== GE graph (single-op) test: shape=[%ld, %ld], elements=%ld ==\n",
                static_cast<long>(n2), static_cast<long>(n1), static_cast<long>(elementCount));

    // ---- 1. 准备 host 数据 ------------------------------------------------
    std::vector<uint16_t> xHost(static_cast<size_t>(elementCount));
    std::vector<uint16_t> yHost(static_cast<size_t>(elementCount));
    std::vector<uint16_t> zExpect(static_cast<size_t>(elementCount));
    for (int64_t i = 0; i < elementCount; ++i) {
        const float xv = static_cast<float>(i % 97) * 0.125F - 5.0F;
        const float yv = static_cast<float>(i % 31) * 0.25F - 2.0F;
        xHost[static_cast<size_t>(i)] = FloatToFp16(xv);
        yHost[static_cast<size_t>(i)] = FloatToFp16(yv);
        zExpect[static_cast<size_t>(i)] = FloatToFp16(xv + yv);
    }

    // ---- 2. 初始化 GE -----------------------------------------------------
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "0"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("[ERROR] GEInitialize failed\n");
        return -1;
    }

    // ---- 3. 构造单算子图 --------------------------------------------------
    const std::vector<int64_t> shapeDims = {n2, n1};
    ge::Shape shape(shapeDims);
    ge::TensorDesc inputDescX(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    ge::TensorDesc inputDescY(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    ge::TensorDesc outputDescZ(shape, ge::FORMAT_ND, ge::DT_FLOAT16);

    ge::Graph graph("add_custom_graph");

    // 图输入占位算子
    auto dataX = ge::op::Data("data_x");
    dataX.set_attr_index(0);
    dataX.update_input_desc_x(inputDescX);
    auto dataY = ge::op::Data("data_y");
    dataY.set_attr_index(1);
    dataY.update_input_desc_y(inputDescY);

    // 目标算子
    auto addCustom = ge::op::AddCustom("add_custom_0");
    addCustom.set_input_x(dataX);
    addCustom.set_input_y(dataY);
    addCustom.update_output_desc_z(outputDescZ);

    graph.AddOp(dataX);
    graph.AddOp(dataY);
    graph.AddOp(addCustom);
    graph.SetInputs({dataX, dataY}).SetOutputs({addCustom});

    // ---- 4. 建 session、跑图 ----------------------------------------------
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);

    const uint32_t graphId = 0;
    if (session.AddGraph(graphId, graph) != ge::SUCCESS) {
        std::printf("[ERROR] AddGraph failed\n");
        ge::GEFinalize();
        return -1;
    }

    ge::Tensor xTensor(inputDescX);
    ge::Tensor yTensor(inputDescY);
    if (xTensor.SetData(reinterpret_cast<uint8_t*>(xHost.data()), byteSize) != ge::SUCCESS ||
        yTensor.SetData(reinterpret_cast<uint8_t*>(yHost.data()), byteSize) != ge::SUCCESS) {
        std::printf("[ERROR] Tensor::SetData failed\n");
        ge::GEFinalize();
        return -1;
    }

    std::vector<ge::Tensor> inputTensors = {xTensor, yTensor};
    std::vector<ge::Tensor> outputTensors;
    if (session.RunGraph(graphId, inputTensors, outputTensors) != ge::SUCCESS) {
        std::printf("[ERROR] RunGraph failed\n");
        ge::GEFinalize();
        return -1;
    }

    // ---- 5. 取结果比对 ----------------------------------------------------
    if (outputTensors.empty()) {
        std::printf("[ERROR] empty graph output\n");
        ge::GEFinalize();
        return -1;
    }

    const uint8_t* outData = outputTensors[0].GetData();
    std::vector<uint16_t> zHost(static_cast<size_t>(elementCount));
    std::memcpy(zHost.data(), outData, byteSize);

    size_t mismatchCount = 0U;
    const double maxErr = CompareFp16(zHost, zExpect, mismatchCount);

    std::printf("--------------------------------------------------\n");
    std::printf("elements      : %ld\n", static_cast<long>(elementCount));
    std::printf("mismatch      : %zu\n", mismatchCount);
    std::printf("max abs error : %.6f\n", maxErr);
    std::printf("result        : %s\n", (mismatchCount == 0U) ? "PASS" : "FAIL");
    std::printf("--------------------------------------------------\n");

    ge::GEFinalize();
    return (mismatchCount == 0U) ? 0 : 1;
}
