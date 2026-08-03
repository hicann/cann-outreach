/*!
 * \file test_aclnn_sync_batch_norm_backward_reduce.cpp
 * \brief SyncBatchNormBackwardReduce 算子 aclnn 端到端测试
 *
 * 支持 3 种数据类型 (float16/float32/bfloat16)，包含 golden 计算与结果验证。
 * 用法:
 *   ./test_aclnn_sync_batch_norm_backward_reduce [float16|float32|bfloat16|all]
 *   默认测试 all (全部 dtype)
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include "acl/acl.h"
#include "aclnn_sync_batch_norm_backward_reduce.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ============================================================
// 辅助函数
// ============================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static float HalfToFloat(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x3ff);
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        exp = 1;
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        mant &= 0x3ff;
        uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
        float result;
        memcpy(&result, &f, sizeof(float));
        return result;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

static float BFloat16ToFloat(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// ============================================================
// 初始化 ACL 环境
// ============================================================

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// ============================================================
// 创建 ACL Tensor (支持 float16/float32/bfloat16)
// ============================================================

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    size_t elemSize = 4;  // default float32
    if (dataType == aclDataType::ACL_FLOAT16 || dataType == aclDataType::ACL_BF16) {
        elemSize = 2;
    }
    auto size = static_cast<int64_t>(elemCount * elemSize);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 将 host 数据转换为目标 dtype 的字节流
    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else {
        // ACL_FLOAT (float32)
        for (int64_t i = 0; i < elemCount; i++) {
            float f = static_cast<float>(hostData[i]);
            memcpy(convBuf.data() + i * 4, &f, 4);
        }
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// ============================================================
// 从设备读取输出并解码为 float32
// ============================================================

std::vector<float> ReadOutputFromDevice(void* deviceAddr, int64_t elemCount, aclDataType dataType)
{
    size_t elemSize = (dataType == aclDataType::ACL_FLOAT) ? 4 : 2;
    size_t totalBytes = static_cast<size_t>(elemCount) * elemSize;
    std::vector<uint8_t> buf(totalBytes);
    auto ret = aclrtMemcpy(buf.data(), totalBytes, deviceAddr, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return {});

    std::vector<float> result(elemCount);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h;
            memcpy(&h, buf.data() + i * 2, 2);
            result[i] = HalfToFloat(h);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b;
            memcpy(&b, buf.data() + i * 2, 2);
            result[i] = BFloat16ToFloat(b);
        }
    } else {
        for (int64_t i = 0; i < elemCount; i++) {
            memcpy(&result[i], buf.data() + i * 4, 4);
        }
    }
    return result;
}

// ============================================================
// 计算 golden 数据
//   dy_mean      = mean * sum_dy
//   sum_dy_xmu   = sum_dy_dx_pad - dy_mean
//   y            = sum_dy_xmu * invert_std
//
// 对于 float16/bfloat16，输入数据经过 CreateAclTensor 已被量化到目标 dtype，
// kernel 实际看到的是量化后的值。因此 golden 计算也需要先量化输入，
// 再在 float32 下计算，最后将结果量化回目标 dtype，与 kernel 行为完全对齐。
// ============================================================

// 将 float32 值量化到目标 dtype 再转回 float32（模拟 kernel 的 Cast 到 float32）
static float QuantizeToDtypeAndBack(float val, aclDataType dataType)
{
    if (dataType == aclDataType::ACL_FLOAT16) {
        uint16_t h = FloatToHalf(val);
        return HalfToFloat(h);
    } else if (dataType == aclDataType::ACL_BF16) {
        uint16_t b = FloatToBFloat16(val);
        return BFloat16ToFloat(b);
    }
    return val;  // float32 无需量化
}

void ComputeGolden(const std::vector<float>& sum_dy, const std::vector<float>& sum_dy_dx_pad,
                   const std::vector<float>& mean, const std::vector<float>& invert_std,
                   std::vector<float>& golden_sum_dy_xmu, std::vector<float>& golden_y,
                   aclDataType dataType)
{
    int64_t n = static_cast<int64_t>(sum_dy.size());
    golden_sum_dy_xmu.resize(n);
    golden_y.resize(n);
    for (int64_t i = 0; i < n; i++) {
        // 先将输入量化到目标 dtype（模拟 kernel 读取 float16/bf16 输入后 cast 到 float32）
        float sd = QuantizeToDtypeAndBack(sum_dy[i], dataType);
        float sddp = QuantizeToDtypeAndBack(sum_dy_dx_pad[i], dataType);
        float mn = QuantizeToDtypeAndBack(mean[i], dataType);
        float isv = QuantizeToDtypeAndBack(invert_std[i], dataType);

        // float32 计算
        float dy_mean = mn * sd;
        float sum_dy_xmu_f32 = sddp - dy_mean;
        float y_f32 = sum_dy_xmu_f32 * isv;

        // 将结果量化回目标 dtype（模拟 kernel 的 Cast 回 float16/bf16）
        golden_sum_dy_xmu[i] = QuantizeToDtypeAndBack(sum_dy_xmu_f32, dataType);
        golden_y[i] = QuantizeToDtypeAndBack(y_f32, dataType);
    }
}

// ============================================================
// 验证结果
// ============================================================

bool VerifyResult(const std::vector<float>& actual, const std::vector<float>& golden,
                  aclDataType dataType, int64_t printCount)
{
    int64_t n = static_cast<int64_t>(actual.size());
    float tolerance = 0.0f;
    if (dataType == aclDataType::ACL_FLOAT16) {
        tolerance = 1e-3f;   // 2^-10
    } else if (dataType == aclDataType::ACL_BF16) {
        tolerance = 1e-2f;   // 2^-7
    } else {
        tolerance = 1e-5f;   // float32
    }

    int64_t failCount = 0;
    int64_t printLimit = std::min(printCount, n);
    LOG_PRINT("  Index |   Actual    |   Golden    |  Diff\n");
    LOG_PRINT("  ------+-------------+-------------+---------\n");
    for (int64_t i = 0; i < n; i++) {
        float diff = std::abs(actual[i] - golden[i]);
        // 使用相对误差 + 绝对误差的混合容差
        float relTol = tolerance * (1.0f + std::abs(golden[i]));
        if (diff > relTol) {
            failCount++;
            if (failCount <= 10) {
                LOG_PRINT("  [%4ld] | %11.6f | %11.6f | %9.2e FAIL\n", i, actual[i], golden[i], diff);
            }
        }
        if (i < printLimit) {
            LOG_PRINT("  [%4ld] | %11.6f | %11.6f | %9.2e %s\n", i, actual[i], golden[i], diff,
                      diff > relTol ? "FAIL" : "ok");
        }
    }
    LOG_PRINT("  ------+-------------+-------------+---------\n");
    LOG_PRINT("  Total: %ld elements, %ld failed, tolerance=%.2e\n", n, failCount, tolerance);
    return failCount == 0;
}

// ============================================================
// 运行单个 dtype 的测试
// ============================================================

int RunTest(const std::string& dtypeStr, aclrtStream stream)
{
    LOG_PRINT("\n========== Testing dtype: %s ==========\n", dtypeStr.c_str());

    aclDataType dataType;
    if (dtypeStr == "float16") {
        dataType = aclDataType::ACL_FLOAT16;
    } else if (dtypeStr == "float32") {
        dataType = aclDataType::ACL_FLOAT;
    } else if (dtypeStr == "bfloat16") {
        dataType = aclDataType::ACL_BF16;
    } else {
        LOG_PRINT("Unknown dtype: %s\n", dtypeStr.c_str());
        return -1;
    }

    // 使用多样化的测试数据 (非全1)，shape 选择多核场景
    std::vector<int64_t> dataShape = {1024};
    int64_t elemCount = GetShapeSize(dataShape);

    // 生成有区分度的输入数据
    std::vector<float> sum_dyHostData(elemCount);
    std::vector<float> sum_dy_dx_padHostData(elemCount);
    std::vector<float> meanHostData(elemCount);
    std::vector<float> invert_stdHostData(elemCount);
    for (int64_t i = 0; i < elemCount; i++) {
        sum_dyHostData[i] = 0.1f * static_cast<float>(i + 1);
        sum_dy_dx_padHostData[i] = 0.5f * static_cast<float>(i + 1);
        meanHostData[i] = 0.2f * static_cast<float>(i % 100 + 1);
        invert_stdHostData[i] = 0.3f + 0.01f * static_cast<float>(i % 50);
    }

    // 创建输入 tensor
    aclTensor* sum_dy = nullptr;
    void* sum_dyDeviceAddr = nullptr;
    aclTensor* sum_dy_dx_pad = nullptr;
    void* sum_dy_dx_padDeviceAddr = nullptr;
    aclTensor* mean = nullptr;
    void* meanDeviceAddr = nullptr;
    aclTensor* invert_std = nullptr;
    void* invert_stdDeviceAddr = nullptr;

    auto ret = CreateAclTensor(sum_dyHostData, dataShape, &sum_dyDeviceAddr, dataType, &sum_dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sum_dy_dx_padHostData, dataShape, &sum_dy_dx_padDeviceAddr, dataType, &sum_dy_dx_pad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, dataShape, &meanDeviceAddr, dataType, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invert_stdHostData, dataShape, &invert_stdDeviceAddr, dataType, &invert_std);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 tensor
    aclTensor* sum_dy_xmu = nullptr;
    void* sum_dy_xmuDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<float> zeroData(elemCount, 0.0f);
    ret = CreateAclTensor(zeroData, dataShape, &sum_dy_xmuDeviceAddr, dataType, &sum_dy_xmu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zeroData, dataShape, &yDeviceAddr, dataType, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一段接口获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSyncBatchNormBackwardReduceGetWorkspaceSize(
        sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("GetWorkspaceSize failed. ERROR: %d, msg: %s\n", ret, errMsg ? errMsg : "null");
        return ret;
    }

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口执行算子
    ret = aclnnSyncBatchNormBackwardReduce(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnSyncBatchNormBackwardReduce failed. ERROR: %d, msg: %s\n", ret, errMsg ? errMsg : "null");
        return ret;
    }

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 读取输出结果
    std::vector<float> actual_sum_dy_xmu = ReadOutputFromDevice(sum_dy_xmuDeviceAddr, elemCount, dataType);
    std::vector<float> actual_y = ReadOutputFromDevice(yDeviceAddr, elemCount, dataType);
    CHECK_RET(!actual_sum_dy_xmu.empty() && !actual_y.empty(), return -1);

    // 计算 golden (与 kernel 行为对齐：先量化输入到目标dtype，float32计算，再量化回目标dtype)
    std::vector<float> golden_sum_dy_xmu, golden_y;
    ComputeGolden(sum_dyHostData, sum_dy_dx_padHostData, meanHostData, invert_stdHostData,
                  golden_sum_dy_xmu, golden_y, dataType);

    // 验证结果
    LOG_PRINT("\n--- Output[0] sum_dy_xmu (first 10 elements) ---\n");
    bool pass1 = VerifyResult(actual_sum_dy_xmu, golden_sum_dy_xmu, dataType, 10);
    LOG_PRINT("  sum_dy_xmu: %s\n\n", pass1 ? "PASSED" : "FAILED");

    LOG_PRINT("--- Output[1] y (first 10 elements) ---\n");
    bool pass2 = VerifyResult(actual_y, golden_y, dataType, 10);
    LOG_PRINT("  y: %s\n", pass2 ? "PASSED" : "FAILED");

    LOG_PRINT("\n========== %s: %s ==========\n",
              dtypeStr.c_str(),
              (pass1 && pass2) ? "ALL PASSED" : "FAILED");

    // 释放资源
    aclDestroyTensor(sum_dy);
    aclrtFree(sum_dyDeviceAddr);
    aclDestroyTensor(sum_dy_dx_pad);
    aclrtFree(sum_dy_dx_padDeviceAddr);
    aclDestroyTensor(mean);
    aclrtFree(meanDeviceAddr);
    aclDestroyTensor(invert_std);
    aclrtFree(invert_stdDeviceAddr);
    aclDestroyTensor(sum_dy_xmu);
    aclrtFree(sum_dy_xmuDeviceAddr);
    aclDestroyTensor(y);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }

    return (pass1 && pass2) ? 0 : -1;
}

// ============================================================
// 快速验证 (不打印详细对比，用于性能测试)
// ============================================================

bool QuickVerify(const std::vector<float>& actual, const std::vector<float>& golden, aclDataType dataType)
{
    int64_t n = static_cast<int64_t>(actual.size());
    float tolerance = 0.0f;
    if (dataType == aclDataType::ACL_FLOAT16) {
        tolerance = 1e-3f;
    } else if (dataType == aclDataType::ACL_BF16) {
        tolerance = 1e-2f;
    } else {
        tolerance = 1e-5f;
    }
    for (int64_t i = 0; i < n; i++) {
        float diff = std::abs(actual[i] - golden[i]);
        float relTol = tolerance * (1.0f + std::abs(golden[i]));
        if (diff > relTol) return false;
    }
    return true;
}

std::string FormatShape(const std::vector<int64_t>& shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

// ============================================================
// 性能测试 (ACL 事件计时)
// ============================================================

int RunPerfTest(const std::string& dtypeStr, const std::vector<int64_t>& dataShape, aclrtStream stream)
{
    aclDataType dataType;
    if (dtypeStr == "float16") {
        dataType = aclDataType::ACL_FLOAT16;
    } else if (dtypeStr == "float32") {
        dataType = aclDataType::ACL_FLOAT;
    } else if (dtypeStr == "bfloat16") {
        dataType = aclDataType::ACL_BF16;
    } else {
        return -1;
    }

    std::string shapeStr = FormatShape(dataShape);
    int64_t elemCount = GetShapeSize(dataShape);

    // 资源指针 (初始化为 nullptr 以便安全清理)
    aclTensor* sum_dy = nullptr;
    void* sum_dyDeviceAddr = nullptr;
    aclTensor* sum_dy_dx_pad = nullptr;
    void* sum_dy_dx_padDeviceAddr = nullptr;
    aclTensor* mean = nullptr;
    void* meanDeviceAddr = nullptr;
    aclTensor* invert_std = nullptr;
    void* invert_stdDeviceAddr = nullptr;
    aclTensor* sum_dy_xmu = nullptr;
    void* sum_dy_xmuDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;

    do {
        // 生成输入数据 (与 TBE 测试使用相同数据，确保公平对比)
        std::vector<float> sum_dyHostData(elemCount);
        std::vector<float> sum_dy_dx_padHostData(elemCount);
        std::vector<float> meanHostData(elemCount);
        std::vector<float> invert_stdHostData(elemCount);
        for (int64_t i = 0; i < elemCount; i++) {
            sum_dyHostData[i] = 0.1f * static_cast<float>(i + 1);
            sum_dy_dx_padHostData[i] = 0.5f * static_cast<float>(i + 1);
            meanHostData[i] = 0.2f * static_cast<float>(i % 100 + 1);
            invert_stdHostData[i] = 0.3f + 0.01f * static_cast<float>(i % 50);
        }

        // 创建输入 tensor
        auto ret = CreateAclTensor(sum_dyHostData, dataShape, &sum_dyDeviceAddr, dataType, &sum_dy);
        if (ret != ACL_SUCCESS) { printf("[ASCEND_C ERROR] %s %s create tensor failed\n", dtypeStr.c_str(), shapeStr.c_str()); break; }
        ret = CreateAclTensor(sum_dy_dx_padHostData, dataShape, &sum_dy_dx_padDeviceAddr, dataType, &sum_dy_dx_pad);
        if (ret != ACL_SUCCESS) { break; }
        ret = CreateAclTensor(meanHostData, dataShape, &meanDeviceAddr, dataType, &mean);
        if (ret != ACL_SUCCESS) { break; }
        ret = CreateAclTensor(invert_stdHostData, dataShape, &invert_stdDeviceAddr, dataType, &invert_std);
        if (ret != ACL_SUCCESS) { break; }

        // 创建输出 tensor
        std::vector<float> zeroData(elemCount, 0.0f);
        ret = CreateAclTensor(zeroData, dataShape, &sum_dy_xmuDeviceAddr, dataType, &sum_dy_xmu);
        if (ret != ACL_SUCCESS) { break; }
        ret = CreateAclTensor(zeroData, dataShape, &yDeviceAddr, dataType, &y);
        if (ret != ACL_SUCCESS) { break; }

        // 获取 workspace + executor
        ret = aclnnSyncBatchNormBackwardReduceGetWorkspaceSize(
            sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            printf("[ASCEND_C ERROR] %s %s GetWorkspaceSize failed: %d, msg: %s\n",
                   dtypeStr.c_str(), shapeStr.c_str(), ret, errMsg ? errMsg : "null");
            break;
        }

        // 申请 workspace
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) { break; }
        }

        // 预热 (运行一次，触发懒加载/初始化)
        ret = aclnnSyncBatchNormBackwardReduce(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            printf("[ASCEND_C ERROR] %s %s warmup failed: %d\n", dtypeStr.c_str(), shapeStr.c_str(), ret);
            break;
        }
        aclrtSynchronizeStream(stream);

        // 创建 ACL 事件用于计时
        aclrtCreateEvent(&startEvent);
        aclrtCreateEvent(&endEvent);

        // 计时运行 (重复 10 次取平均)
        {
            int repeat = 10;
            aclrtRecordEvent(startEvent, stream);
            for (int i = 0; i < repeat; i++) {
                aclnnSyncBatchNormBackwardReduce(workspaceAddr, workspaceSize, executor, stream);
            }
            aclrtRecordEvent(endEvent, stream);
            aclrtSynchronizeStream(stream);

            float elapsedTimeMs = 0.0f;
            aclrtEventElapsedTime(&elapsedTimeMs, startEvent, endEvent);
            float avgUs = (elapsedTimeMs / repeat) * 1000.0f;

            printf("[ASCEND_C PERF] SyncBatchNormBackwardReduce %s shape=%s elapsed_us=%.2f\n",
                   dtypeStr.c_str(), shapeStr.c_str(), avgUs);
        }

        // 验证结果
        {
            std::vector<float> actual_xmu = ReadOutputFromDevice(sum_dy_xmuDeviceAddr, elemCount, dataType);
            std::vector<float> actual_y = ReadOutputFromDevice(yDeviceAddr, elemCount, dataType);
            std::vector<float> golden_xmu, golden_y;
            ComputeGolden(sum_dyHostData, sum_dy_dx_padHostData, meanHostData, invert_stdHostData,
                          golden_xmu, golden_y, dataType);
            bool pass = QuickVerify(actual_xmu, golden_xmu, dataType) && QuickVerify(actual_y, golden_y, dataType);
            if (pass) {
                printf("[ASCEND_C PASS] SyncBatchNormBackwardReduce %s shape=%s\n",
                       dtypeStr.c_str(), shapeStr.c_str());
            } else {
                printf("[ASCEND_C FAIL] SyncBatchNormBackwardReduce %s shape=%s\n",
                       dtypeStr.c_str(), shapeStr.c_str());
            }
        }
    } while (0);

    // 清理资源
    if (endEvent) aclrtDestroyEvent(endEvent);
    if (startEvent) aclrtDestroyEvent(startEvent);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (executor) aclDestroyAclOpExecutor(executor);
    if (sum_dy) { aclDestroyTensor(sum_dy); aclrtFree(sum_dyDeviceAddr); }
    if (sum_dy_dx_pad) { aclDestroyTensor(sum_dy_dx_pad); aclrtFree(sum_dy_dx_padDeviceAddr); }
    if (mean) { aclDestroyTensor(mean); aclrtFree(meanDeviceAddr); }
    if (invert_std) { aclDestroyTensor(invert_std); aclrtFree(invert_stdDeviceAddr); }
    if (sum_dy_xmu) { aclDestroyTensor(sum_dy_xmu); aclrtFree(sum_dy_xmuDeviceAddr); }
    if (y) { aclDestroyTensor(y); aclrtFree(yDeviceAddr); }

    return 0;
}

// ============================================================
// 主函数
// ============================================================

int main(int argc, char** argv)
{
    std::string mode = (argc > 1) ? argv[1] : "all";

    // 初始化 ACL
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (mode == "perf") {
        // 性能测试模式
        struct PerfCase { std::string dtype; std::vector<int64_t> shape; };
        std::vector<PerfCase> perfCases = {
            {"float32", {1}},
            {"float16", {1}},
            {"bfloat16", {1}},
            {"float32", {100}},
            {"float16", {100}},
            {"float32", {1024}},
            {"float16", {1024}},
            {"bfloat16", {1024}},
            {"float32", {65536}},
            {"float16", {65536}},
            {"bfloat16", {65536}},
            {"float16", {4, 1024}},
            {"float32", {2, 4, 1024}},
        };

        LOG_PRINT("\n========== Ascend C Performance Test ==========\n");
        for (const auto& tc : perfCases) {
            RunPerfTest(tc.dtype, tc.shape, stream);
        }
        LOG_PRINT("\n========== Performance Test Complete ==========\n");
    } else {
        // 功能测试模式 (原有行为)
        std::vector<std::string> dtypesToTest;
        if (mode == "all") {
            dtypesToTest = {"float16", "float32", "bfloat16"};
        } else {
            dtypesToTest = {mode};
        }

        int totalPass = 0;
        int totalFail = 0;
        for (const auto& dt : dtypesToTest) {
            int result = RunTest(dt, stream);
            if (result == 0) {
                totalPass++;
            } else {
                totalFail++;
            }
        }

        LOG_PRINT("\n============================================================\n");
        LOG_PRINT("Test Summary: %d passed, %d failed (total %d)\n",
                  totalPass, totalFail, totalPass + totalFail);
        LOG_PRINT("============================================================\n");
    }

    // 释放资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
