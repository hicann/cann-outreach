/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/truncate_mod_proto.h"

#define FAILED -1
#define SUCCESS 0
#define WARMUP_ITERATIONS 10
#define TEST_ITERATIONS 100

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

namespace {
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    struct tm tm_info;
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tm_info));
    return tmp;
}

template <typename T>
ge::Tensor CreateTensor(vector<int64_t> shape, ge::DataType d_type, const vector<T>& values)
{
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, d_type);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= static_cast<size_t>(shape[i]);
    }
    if (values.size() != size) {
        printf("%s - ERROR - Generate input tensor failed\n", GetTime().c_str());
        return ge::Tensor{};
    }

    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

ge::Operator CreateInputNode(size_t i, TensorDesc desc)
{
    auto node = op::Data("placeholder" + std::to_string(i)).set_attr_index(i);
    node.update_input_desc_x(desc);
    node.update_output_desc_y(desc);
    return node;
}

// 测试张量场景
int TestTensorScenario(size_t dataSize, double& durationMs)
{
    ge::DataType d_type = ge::DT_FLOAT;
    auto node_truncate_mod = op::TruncateMod("truncate_mod");
    vector<int64_t> shape = {1, static_cast<int64_t>(dataSize)};

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    std::uniform_real_distribution<float> disOther(0.1f, 10.0f);

    vector<float> x1Data(dataSize);
    vector<float> x2Data(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        x1Data[i] = dis(gen);
        x2Data[i] = disOther(gen);
    }

    auto tensor_x1 = CreateTensor<float>(shape, d_type, x1Data);
    auto tensor_x2 = CreateTensor<float>(shape, d_type, x2Data);
    if (tensor_x1.GetSize() == 0 || tensor_x2.GetSize() == 0) {
        return FAILED;
    }

    auto node_x1 = CreateInputNode(0, tensor_x1.GetTensorDesc());
    auto node_x2 = CreateInputNode(1, tensor_x2.GetTensorDesc());

    vector<ge::Tensor> input = {tensor_x1, tensor_x2};
    vector<Operator> inputs = {node_x1, node_x2};

    Graph graph("truncate_mod_tensor_test");
    graph.AddOp(node_x1);
    graph.AddOp(node_x2);
    node_truncate_mod.set_input_x1(node_x1);
    node_truncate_mod.set_input_x2(node_x2);

    TensorDesc y_desc(ge::Shape(shape), FORMAT_ND, d_type);
    node_truncate_mod.update_output_desc_y(y_desc);
    vector<Operator> outputs = {node_truncate_mod};

    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options;
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));
    if (session == nullptr) {
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ge::Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        return FAILED;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        vector<ge::Tensor> output;
        session->RunGraph(graph_id, input, output);
    }

    // Performance test
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        vector<ge::Tensor> output;
        session->RunGraph(graph_id, input, output);
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count() / TEST_ITERATIONS;

    return SUCCESS;
}

// 测试标量场景
int TestScalarScenario(size_t dataSize, double& durationMs)
{
    ge::DataType d_type = ge::DT_FLOAT;
    auto node_truncate_mod = op::TruncateMod("truncate_mod");
    vector<int64_t> x1Shape = {1, static_cast<int64_t>(dataSize)};
    vector<int64_t> x2Shape = {1, 1}; // 标量

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    vector<float> x1Data(dataSize);
    vector<float> x2Data = {3.14f}; // 标量
    for (size_t i = 0; i < dataSize; ++i) {
        x1Data[i] = dis(gen);
    }

    auto tensor_x1 = CreateTensor<float>(x1Shape, d_type, x1Data);
    auto tensor_x2 = CreateTensor<float>(x2Shape, d_type, x2Data);
    if (tensor_x1.GetSize() == 0 || tensor_x2.GetSize() == 0) {
        return FAILED;
    }

    auto node_x1 = CreateInputNode(0, tensor_x1.GetTensorDesc());
    auto node_x2 = CreateInputNode(1, tensor_x2.GetTensorDesc());

    vector<ge::Tensor> input = {tensor_x1, tensor_x2};
    vector<Operator> inputs = {node_x1, node_x2};

    Graph graph("truncate_mod_scalar_test");
    graph.AddOp(node_x1);
    graph.AddOp(node_x2);
    node_truncate_mod.set_input_x1(node_x1);
    node_truncate_mod.set_input_x2(node_x2);

    TensorDesc y_desc(ge::Shape(x1Shape), FORMAT_ND, d_type);
    node_truncate_mod.update_output_desc_y(y_desc);
    vector<Operator> outputs = {node_truncate_mod};

    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options;
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));
    if (session == nullptr) {
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ge::Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        return FAILED;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        vector<ge::Tensor> output;
        session->RunGraph(graph_id, input, output);
    }

    // Performance test
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        vector<ge::Tensor> output;
        session->RunGraph(graph_id, input, output);
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count() / TEST_ITERATIONS;

    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    printf("TruncateMod Performance Test\n");
    printf("============================\n\n");

    const char* graph_name = "truncate_mod_perf_test";
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ge::Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - Initialize GE failed\n", GetTime().c_str());
        return FAILED;
    }

    // Test different data sizes
    vector<size_t> testSizes = {1024, 4096, 16384, 65536, 262144, 1048576};

    printf("Testing Tensor Scenario:\n");
    printf("Data Size | Duration (ms) | Throughput (GB/s)\n");
    printf("----------|---------------|------------------\n");

    for (size_t size : testSizes) {
        double durationMs = 0;
        if (TestTensorScenario(size, durationMs) == SUCCESS) {
            double throughputGBps = (size * sizeof(float) * 2) / (durationMs * 1e6); // 读写各一次
            printf("%8zu | %13.3f | %16.3f\n", size, durationMs, throughputGBps);
        } else {
            printf("%8zu | FAILED        | -\n", size);
        }
    }

    printf("\nTesting Scalar Scenario:\n");
    printf("Data Size | Duration (ms) | Throughput (GB/s)\n");
    printf("----------|---------------|------------------\n");

    for (size_t size : testSizes) {
        double durationMs = 0;
        if (TestScalarScenario(size, durationMs) == SUCCESS) {
            double throughputGBps = (size * sizeof(float) * 2) / (durationMs * 1e6);
            printf("%8zu | %13.3f | %16.3f\n", size, durationMs, throughputGBps);
        } else {
            printf("%8zu | FAILED        | -\n", size);
        }
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }

    printf("\nPerformance test completed.\n");
    return SUCCESS;
}