#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/op_errno.h" // ACLNN_SUCCESS 宏定义
#include "aclnn_truncate_mod.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

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

// ---- FP16 / BF16 编解码（无浮点硬件依赖，主机侧纯位运算） ----

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // ±0
        } else {
            // 次正规数规范化（本示例数据 |x|>=0.5 不会触发，保留处理）
            int32_t e = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13); // Inf/NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

static float BFloat16ToFloat(uint16_t b)
{
    uint32_t bits = (uint32_t)b << 16;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

// ---- 右对齐广播索引映射（numpy 语义） ----
// 对每个输出线性索引，计算其对应的输入线性索引；广播维度（输入 dim==1）映射到 0。
static void BuildIndexMap(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                          std::vector<int64_t>& idxMap)
{
    const int64_t total = GetShapeSize(outShape);
    const int64_t inRank = static_cast<int64_t>(inShape.size());
    const int64_t outRank = static_cast<int64_t>(outShape.size());
    idxMap.assign(total, 0);
    if (GetShapeSize(inShape) == 1) {
        return; // 标量广播：全部映射到 0
    }
    // inStrides[k]：输出第 k 维（从右往左）对输入线性索引的贡献步长；广播维为 0
    std::vector<int64_t> inStrides(outRank, 1);
    int64_t dim = 1;
    for (int64_t k = outRank - 1; k >= 0; k--) {
        int64_t dimIn = 1;
        if (k >= outRank - inRank) {
            dimIn = inShape[k - (outRank - inRank)];
        }
        inStrides[k] = (dimIn > 1) ? dim : 0;
        dim *= outShape[k];
    }
    for (int64_t i = 0; i < total; i++) {
        int64_t idx = i;
        int64_t inIdx = 0;
        for (int64_t k = outRank - 1; k >= 0; k--) {
            int64_t coord = idx % outShape[k];
            idx /= outShape[k];
            inIdx += coord * inStrides[k];
        }
        idxMap[i] = inIdx;
    }
}

// ---- 用例定义 ----

struct AclnnCase {
    const char* name;             // A1/A2/A3
    std::vector<int64_t> x1Shape;
    std::vector<int64_t> x2Shape;
    aclDataType dtype;
    double rtol;                  // ops-precision-standard 混合容差
    double atol;
    double maxAbsLimit;           // max_abs_error_limit
    uint32_t seed;
};

static const AclnnCase g_cases[] = {
    {"A1 FP16 [16]x[16]", {16}, {16}, ACL_FLOAT16, 2.0 / 512.0, 2.0 / 512.0, 0.1, 101},
    {"A2 BF16 [16]x[16]", {16}, {16}, ACL_BF16, 1.0 / 64.0, 1.0 / 64.0, 1.0, 202},
    {"A3 FP32 [4,8]x[8] broadcast", {4, 8}, {8}, ACL_FLOAT, 1.0 / 1024.0, 1.0 / 65536.0, 0.01, 303},
};

std::string ShapeStr(const std::vector<int64_t>& shape); // 前向声明（RunCase 中使用）

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

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
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
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

// 读取设备输出到主机并按 dtype 解码为 float 数组
static int ReadBackTensor(aclrtStream stream, void* deviceAddr, size_t bytes, aclDataType dtype,
                          int64_t elemCount, std::vector<float>& outFloat)
{
    std::vector<uint8_t> hostBuf(bytes);
    auto ret = aclrtMemcpy(hostBuf.data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy(D2H) failed. ERROR: %d\n", ret); return ret);

    outFloat.resize(elemCount);
    if (dtype == ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h;
            memcpy(&h, hostBuf.data() + i * 2, 2);
            outFloat[i] = HalfToFloat(h);
        }
    } else if (dtype == ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b;
            memcpy(&b, hostBuf.data() + i * 2, 2);
            outFloat[i] = BFloat16ToFloat(b);
        }
    } else {
        for (int64_t i = 0; i < elemCount; i++) {
            memcpy(&outFloat[i], hostBuf.data() + i * 4, 4);
        }
    }
    return 0;
}

// FP64 golden：y = x1 - trunc(x1/x2) * x2（右对齐广播）
static std::vector<double> ComputeGolden(const std::vector<float>& x1F, const std::vector<float>& x2F,
                                         const AclnnCase& c)
{
    std::vector<int64_t> idxX1, idxX2;
    std::vector<int64_t> yShape = c.x1Shape;
    for (size_t k = 0; k < c.x2Shape.size(); k++) {
        int64_t idx = static_cast<int64_t>(yShape.size()) - static_cast<int64_t>(c.x2Shape.size()) + k;
        int64_t dOut = yShape[idx];
        int64_t dIn = c.x2Shape[k];
        if (dOut == 1) {
            yShape[idx] = dIn;
        } else if (dIn != dOut) {
            std::cerr << "invalid broadcast shape" << std::endl;
            std::exit(1);
        }
    }
    BuildIndexMap(c.x1Shape, yShape, idxX1);
    BuildIndexMap(c.x2Shape, yShape, idxX2);
    const int64_t total = GetShapeSize(yShape);
    std::vector<double> golden(total);
    for (int64_t i = 0; i < total; i++) {
        double a = static_cast<double>(x1F[idxX1[i]]);
        double b = static_cast<double>(x2F[idxX2[i]]);
        golden[i] = a - std::trunc(a / b) * b;
    }
    return golden;
}

// 按 ops-precision-standard 混合容差逐元素比对
static bool CompareOutput(const std::vector<float>& actual, const std::vector<double>& golden,
                          const AclnnCase& c, int64_t total)
{
    int64_t matched = 0;
    double maxAbsError = 0.0;
    double sumRel = 0.0, maxRel = 0.0;
    int64_t finiteCnt = 0;
    int64_t worstIdx = 0;
    double worstDiff = -1.0;
    for (int64_t i = 0; i < total; i++) {
        double a = static_cast<double>(actual[i]);
        double g = golden[i];
        double diff = std::fabs(a - g);
        if (diff > worstDiff) {
            worstDiff = diff;
            worstIdx = i;
        }
        if (diff <= c.atol + c.rtol * std::fabs(g)) {
            matched++;
        }
        if (diff > maxAbsError) {
            maxAbsError = diff;
        }
        if (std::isfinite(g) && std::isfinite(a)) {
            double denom = std::max(std::fabs(g), 1e-6);
            double rel = diff / denom;
            sumRel += rel;
            if (rel > maxRel) {
                maxRel = rel;
            }
            finiteCnt++;
        }
    }
    double matchedRatio = static_cast<double>(matched) / static_cast<double>(total);
    double mere = (finiteCnt > 0) ? sumRel / finiteCnt : 0.0;
    double mare = maxRel;
    bool pass = (matchedRatio >= 0.99) && (maxAbsError <= c.maxAbsLimit);
    LOG_PRINT("  matched_ratio=%.6f (req>=0.99), max_abs_error=%.6e (limit=%.6e), "
              "MERE=%.6e, MARE=%.6e\n",
              matchedRatio, maxAbsError, c.maxAbsLimit, mere, mare);
    if (!pass) {
        LOG_PRINT("  worst diff: index=%lld, output=%f, golden=%.10f\n", (long long)worstIdx, actual[worstIdx],
                  golden[worstIdx]);
    }
    return pass;
}

int RunCase(const AclnnCase& c, aclrtStream stream)
{
    const int64_t x1Count = GetShapeSize(c.x1Shape);
    const int64_t x2Count = GetShapeSize(c.x2Shape);
    // 输出 shape = 右对齐广播结果
    std::vector<int64_t> yShape = c.x1Shape;
    for (size_t k = 0; k < c.x2Shape.size(); k++) {
        int64_t idx = static_cast<int64_t>(yShape.size()) - static_cast<int64_t>(c.x2Shape.size()) + k;
        int64_t dOut = yShape[idx];
        int64_t dIn = c.x2Shape[k];
        if (dOut == 1) {
            yShape[idx] = dIn;
        } else if (dIn != dOut) {
            LOG_PRINT("invalid broadcast shape\n");
            return 1;
        }
    }
    const int64_t yCount = GetShapeSize(yShape);

    // 生成输入数据：x1 ~ U(-10,10)；x2 幅值在 [0.5,5] 且非 0（避免商溢出/除零）
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> distX1(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distMag(0.5f, 5.0f);
    std::uniform_int_distribution<int> signChoice(0, 1);
    std::vector<float> x1Host(x1Count), x2Host(x2Count);
    for (int64_t i = 0; i < x1Count; i++) {
        x1Host[i] = distX1(rng);
    }
    for (int64_t i = 0; i < x2Count; i++) {
        float mag = distMag(rng);
        x2Host[i] = (signChoice(rng) == 0) ? mag : -mag;
    }

    LOG_PRINT("  x1=%s(%lld elems), x2=%s(%lld elems) -> y=%s(%lld elems)\n", ShapeStr(c.x1Shape).c_str(),
              (long long)x1Count, ShapeStr(c.x2Shape).c_str(), (long long)x2Count, ShapeStr(yShape).c_str(),
              (long long)yCount);

    // 构造设备 tensor
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* y = nullptr;
    void* x1Dev = nullptr;
    void* x2Dev = nullptr;
    void* yDev = nullptr;
    int ret = CreateAclTensor(x1Host, c.x1Shape, &x1Dev, c.dtype, &x1);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(x2Host, c.x2Shape, &x2Dev, c.dtype, &x2);
    CHECK_RET(ret == 0, return ret);
    std::vector<float> yHost(yCount, 0.0f);
    ret = CreateAclTensor(yHost, yShape, &yDev, c.dtype, &y);
    CHECK_RET(ret == 0, return ret);

    // 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnTruncateModGetWorkspaceSize(x1, x2, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS,
              LOG_PRINT("aclnnTruncateModGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 第二段接口
    ret = aclnnTruncateMod(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("aclnnTruncateMod failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 读回输出
    int64_t elemSize = (c.dtype == ACL_FLOAT) ? 4 : 2;
    size_t yBytes = static_cast<size_t>(yCount) * static_cast<size_t>(elemSize);
    std::vector<float> actual;
    ret = ReadBackTensor(stream, yDev, yBytes, c.dtype, yCount, actual);
    CHECK_RET(ret == 0, return ret);

    // 解码输入为 float 值（FP16/BF16 位 → float），FP64 golden 与核输出比对
    std::vector<float> x1Val(x1Count), x2Val(x2Count);
    if (c.dtype == ACL_FLOAT16) {
        for (int64_t i = 0; i < x1Count; i++) {
            x1Val[i] = HalfToFloat(FloatToHalf(x1Host[i]));
        }
        for (int64_t i = 0; i < x2Count; i++) {
            x2Val[i] = HalfToFloat(FloatToHalf(x2Host[i]));
        }
    } else if (c.dtype == ACL_BF16) {
        for (int64_t i = 0; i < x1Count; i++) {
            x1Val[i] = BFloat16ToFloat(FloatToBFloat16(x1Host[i]));
        }
        for (int64_t i = 0; i < x2Count; i++) {
            x2Val[i] = BFloat16ToFloat(FloatToBFloat16(x2Host[i]));
        }
    } else {
        x1Val = x1Host;
        x2Val = x2Host;
    }
    std::vector<double> golden = ComputeGolden(x1Val, x2Val, c);
    bool pass = CompareOutput(actual, golden, c, yCount);

    // 释放资源
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(y);
    aclrtFree(x1Dev);
    aclrtFree(x2Dev);
    aclrtFree(yDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    return pass ? 0 : 1;
}

std::string ShapeStr(const std::vector<int64_t>& shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

int main()
{
    // ASCEND_VISIBLE_DEVICES 已把物理设备映射为逻辑 0；可用 ACLNN_DEVICE_ID 覆盖
    int32_t deviceId = 0;
    if (const char* envDev = std::getenv("ACLNN_DEVICE_ID")) {
        deviceId = static_cast<int32_t>(std::atoi(envDev));
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("========================================\n");
    LOG_PRINT("truncate_mod aclnn 通路验证（device=%d）\n", deviceId);
    LOG_PRINT("========================================\n");

    int failCnt = 0;
    for (const auto& c : g_cases) {
        LOG_PRINT("[RUN] %s\n", c.name);
        int r = RunCase(c, stream);
        if (r == 0) {
            LOG_PRINT("[PASS] %s\n", c.name);
        } else {
            LOG_PRINT("[FAIL] %s\n", c.name);
            failCnt++;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failCnt == 0) {
        LOG_PRINT("========================================\n");
        LOG_PRINT("全部 aclnn 用例通过\n");
        LOG_PRINT("========================================\n");
        return 0;
    }
    LOG_PRINT("存在 %d 个失败用例\n", failCnt);
    return 1;
}
