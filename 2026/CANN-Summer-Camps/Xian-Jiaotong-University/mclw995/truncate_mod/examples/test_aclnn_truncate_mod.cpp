#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include "acl/acl.h"
#include "aclnn_truncate_mod.h"

#define CHECK_RET(c, m) do { if (!(c)) { printf("FAIL: %s (line %d)\n", m, __LINE__); return 1; } } while(0)

static uint16_t FloatToFP16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((int32_t)((bits >> 23) & 0xFF)) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3FF;
    if ((bits & 0x7FFFFFFF) == 0) return (uint16_t)sign;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) {
        if ((bits & 0x007FFFFF) == 0) return (uint16_t)(sign | 0x7C00);
        return (uint16_t)(sign | 0x7E00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mantissa);
}

static float FP16ToFloat(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    int32_t exp16 = (h >> 10) & 0x1F;
    uint32_t mantissa16 = h & 0x3FF;
    if ((h & 0x7FFF) == 0) {
        float r; memcpy(&r, &sign, 4); return r;
    }
    if (exp16 == 0) {
        int32_t exp = -14 + 127;
        uint32_t mantissa = mantissa16 << 13;
        uint32_t bits = sign | ((uint32_t)exp << 23) | mantissa;
        float r; memcpy(&r, &bits, 4); return r;
    }
    if (exp16 == 31) {
        if (mantissa16 == 0) {
            uint32_t bits = sign | 0x7F800000;
            float r; memcpy(&r, &bits, 4); return r;
        }
        uint32_t bits = sign | 0x7FC00000;
        float r; memcpy(&r, &bits, 4); return r;
    }
    int32_t exp = exp16 - 15 + 127;
    uint32_t mantissa = mantissa16 << 13;
    uint32_t bits = sign | ((uint32_t)exp << 23) | mantissa;
    float r; memcpy(&r, &bits, 4); return r;
}

static uint16_t FloatToBF16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static float BF16ToFloat(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float r; memcpy(&r, &bits, 4); return r;
}

static int64_t ShapeSize(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (auto v : s) n *= v;
    return n;
}

static std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t rd = std::max(a.size(), b.size());
    std::vector<int64_t> out(rd, 1);
    for (size_t i = 0; i < rd; ++i) {
        int64_t sa = (i < rd - a.size()) ? 1 : a[i - (rd - a.size())];
        int64_t sb = (i < rd - b.size()) ? 1 : b[i - (rd - b.size())];
        if (sa != sb && sa != 1 && sb != 1) return {};
        out[i] = std::max(sa, sb);
    }
    return out;
}

static int64_t BroadcastOffset(int64_t flatIdx, const std::vector<int64_t>& srcShape,
                                const std::vector<int64_t>& outShape) {
    int od = (int)outShape.size();
    int64_t multiIdx[8] = {0};
    int64_t tmp = flatIdx;
    for (int i = od - 1; i >= 0; --i) {
        multiIdx[i] = tmp % outShape[i];
        tmp /= outShape[i];
    }
    int sd = (int)srcShape.size();
    int rankDiff = od - sd;
    int64_t srcIdx = 0;
    for (int i = 0; i < sd; ++i) {
        int outDim = i + rankDiff;
        int64_t dim = (srcShape[i] == 1) ? 0 : multiIdx[outDim];
        srcIdx = srcIdx * srcShape[i] + dim;
    }
    return srcIdx;
}

template <typename T>
static int CreateTensor(const std::vector<T>& data, const std::vector<int64_t>& shape,
                         void** devAddr, aclDataType dt, aclTensor** tensor)
{
    int64_t n = ShapeSize(shape);
    if (n == 0) {
        *tensor = aclCreateTensor(nullptr, 0, dt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, nullptr);
        *devAddr = nullptr;
        return 0;
    }
    auto sz = n * sizeof(T);
    auto ret = aclrtMalloc(devAddr, sz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc");
    if (n > 0 && data.size() >= (size_t)n) {
        ret = aclrtMemcpy(*devAddr, sz, data.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, "memcpy H2D");
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
        strides[i] = shape[i + 1] * strides[i + 1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                               ACL_FORMAT_ND, nullptr, 0, *devAddr);
    CHECK_RET(*tensor != nullptr, "create tensor");
    return 0;
}

template <typename T>
static int RunOp(const std::vector<T>& x, const std::vector<int64_t>& xShape,
                 const std::vector<T>& y, const std::vector<int64_t>& yShape,
                 std::vector<T>& out, const std::vector<int64_t>& outShape,
                 aclDataType dt, aclrtStream stream)
{
    void *xd = nullptr, *yd = nullptr, *od = nullptr;
    aclTensor *xt = nullptr, *yt = nullptr, *ot = nullptr;
    if (CreateTensor(x, xShape, &xd, dt, &xt)) { printf("  CreateTensor x failed\n"); return 1; }
    if (CreateTensor(y, yShape, &yd, dt, &yt)) { printf("  CreateTensor y failed\n"); return 1; }

    int64_t no = ShapeSize(outShape);
    out.resize(no);
    std::vector<T> outInit(no, T(0));
    if (CreateTensor(outInit, outShape, &od, dt, &ot)) { printf("  CreateTensor out failed\n"); return 1; }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    auto ret = aclnnTruncateModGetWorkspaceSize(xt, yt, ot, &wsSize, &exec);
    if (ret != 0) {
        printf("  GetWorkspaceSize returned %d\n", ret);
        aclDestroyTensor(xt); aclDestroyTensor(yt); aclDestroyTensor(ot);
        if (xd) aclrtFree(xd); if (yd) aclrtFree(yd); if (od) aclrtFree(od);
        return 1;
    }

    void *ws = nullptr;
    if (wsSize > 0) {
        auto r = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, "ws malloc");
    }
    ret = aclnnTruncateMod(ws, wsSize, exec, stream);
    if (ret != 0) {
        printf("  Execute returned %d\n", ret);
        if (ws) aclrtFree(ws);
        aclDestroyTensor(xt); aclDestroyTensor(yt); aclDestroyTensor(ot);
        if (xd) aclrtFree(xd); if (yd) aclrtFree(yd); if (od) aclrtFree(od);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "sync failed");

    if (no > 0) {
        auto sz = no * sizeof(T);
        ret = aclrtMemcpy(out.data(), sz, od, sz, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, "D2H copy failed");
    }

    aclDestroyTensor(xt); aclDestroyTensor(yt); aclDestroyTensor(ot);
    aclrtFree(xd); aclrtFree(yd); aclrtFree(od);
    if (ws) aclrtFree(ws);
    return 0;
}

static int32_t golden_i32(int32_t x, int32_t y) {
    if (y == 0) return 0;
    float xf = (float)x, yf = (float)y;
    int32_t q = (int32_t)(xf / yf);
    return x - q * y;
}
static uint8_t golden_u8(uint8_t x, uint8_t y) {
    if (y == 0) return 0;
    return x - (uint8_t)((float)x / (float)y) * y;
}
static int8_t golden_i8(int8_t x, int8_t y) {
    if (y == 0) return 0;
    float xf = (float)x, yf = (float)y;
    int8_t q = (int8_t)(xf / yf);
    return x - q * y;
}
static float golden_f32(float x, float y) {
    if (y == 0.0f) return std::numeric_limits<float>::quiet_NaN();
    return x - std::trunc(x / y) * y;
}

template <typename T>
static bool CmpResult(const std::vector<T>& out, const std::vector<T>& expected,
                       int64_t N, float rtol, float atol, int& errCnt);

template <>
bool CmpResult<float>(const std::vector<float>& out, const std::vector<float>& expected,
                       int64_t N, float rtol, float atol, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        float ov = out[i], ev = expected[i];
        bool oNan = std::isnan(ov), eNan = std::isnan(ev);
        if (oNan && eNan) continue;
        if (oNan || eNan) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
            continue;
        }
        float diff = std::fabs(ov - ev);
        bool bothNearZero = (std::fabs(ov) < 1e-6f && std::fabs(ev) < 1e-6f);
        float thresh = atol + rtol * std::max(std::fabs(ev), 1.0f);
        bool ok = bothNearZero || (diff <= thresh);
        if (!ok) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
        }
    }
    return errCnt == 0;
}

template <>
bool CmpResult<int32_t>(const std::vector<int32_t>& out, const std::vector<int32_t>& expected,
                         int64_t N, float, float, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (out[i] != expected[i]) {
            if (errCnt < 5)
                printf("    [%ld] got=%d exp=%d\n", i, out[i], expected[i]);
            errCnt++;
        }
    }
    return errCnt == 0;
}

template <>
bool CmpResult<uint8_t>(const std::vector<uint8_t>& out, const std::vector<uint8_t>& expected,
                         int64_t N, float, float, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (out[i] != expected[i]) {
            if (errCnt < 5)
                printf("    [%ld] got=%u exp=%u\n", i, out[i], expected[i]);
            errCnt++;
        }
    }
    return errCnt == 0;
}

template <>
bool CmpResult<int8_t>(const std::vector<int8_t>& out, const std::vector<int8_t>& expected,
                        int64_t N, float, float, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (out[i] != expected[i]) {
            if (errCnt < 5)
                printf("    [%ld] got=%d exp=%d\n", i, (int)out[i], (int)expected[i]);
            errCnt++;
        }
    }
    return errCnt == 0;
}

template <>
bool CmpResult<uint16_t>(const std::vector<uint16_t>& out, const std::vector<uint16_t>& expected,
                          int64_t N, float rtol, float atol, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        float ov = FP16ToFloat(out[i]);
        float ev = FP16ToFloat(expected[i]);
        bool oNan = std::isnan(ov), eNan = std::isnan(ev);
        if (oNan && eNan) continue;
        if (oNan || eNan) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
            continue;
        }
        float diff = std::fabs(ov - ev);
        bool bothNearZero = (std::fabs(ov) < 1e-3f && std::fabs(ev) < 1e-3f);
        float thresh = atol + rtol * std::max(std::fabs(ev), 1.0f);
        bool ok = bothNearZero || (diff <= thresh);
        if (!ok) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
        }
    }
    return errCnt == 0;
}

static bool CmpResultBF16(const std::vector<uint16_t>& out, const std::vector<uint16_t>& expected,
                           int64_t N, float rtol, float atol, int& errCnt) {
    errCnt = 0;
    for (int64_t i = 0; i < N; ++i) {
        float ov = BF16ToFloat(out[i]);
        float ev = BF16ToFloat(expected[i]);
        bool oNan = std::isnan(ov), eNan = std::isnan(ev);
        if (oNan && eNan) continue;
        if (oNan || eNan) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
            continue;
        }
        float diff = std::fabs(ov - ev);
        bool bothNearZero = (std::fabs(ov) < 1e-2f && std::fabs(ev) < 1e-2f);
        float thresh = atol + rtol * std::max(std::fabs(ev), 1.0f);
        bool ok = bothNearZero || (diff <= thresh);
        if (!ok) {
            if (errCnt < 5)
                printf("    [%ld] got=%f exp=%f\n", i, ov, ev);
            errCnt++;
        }
    }
    return errCnt == 0;
}

template <typename T, typename FloatFunc>
static void ComputeGoldenBroadcast(const std::vector<T>& x, const std::vector<int64_t>& xShape,
                                    const std::vector<T>& y, const std::vector<int64_t>& yShape,
                                    std::vector<T>& out, const std::vector<int64_t>& outShape,
                                    FloatFunc goldenElem)
{
    int64_t N = ShapeSize(outShape);
    out.resize(N);
    for (int64_t i = 0; i < N; ++i) {
        int64_t xOff = BroadcastOffset(i, xShape, outShape);
        int64_t yOff = BroadcastOffset(i, yShape, outShape);
        out[i] = goldenElem(x[xOff], y[yOff]);
    }
}

struct TestCase {
    std::string name;
    std::vector<int64_t> xShape;
    std::vector<int64_t> yShape;
};

static std::vector<TestCase> GetTestCases() {
    return {
        {"same_100", {100}, {100}},
        {"same_1024", {1024}, {1024}},
        {"same_8x64", {8, 64}, {8, 64}},
        {"same_32x128", {32, 128}, {32, 128}},
        {"scalar_broadcast", {100}, {1}},
        {"scalar_broadcast_rev", {1}, {100}},
        {"3x1_vs_1x4", {3, 1}, {1, 4}},
        {"1x4_vs_3x1", {1, 4}, {3, 1}},
        {"1D_vs_2D", {64}, {8, 1}},
        {"2D_vs_1D", {8, 1}, {64}},
        {"non_divisible", {100}, {37}},
        {"large_multi_core", {1024 * 1024}, {1024 * 1024}},
    };
}

static void FillDataI32(std::vector<int32_t>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i) data[i] = (rand() % 2000) - 1000;
}
static void FillDataU8(std::vector<uint8_t>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i) data[i] = (uint8_t)(rand() % 200);
}
static void FillDataI8(std::vector<int8_t>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i) data[i] = (int8_t)((rand() % 200) - 100);
}
static void FillDataF32(std::vector<float>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i) data[i] = ((rand() % 2000) - 1000) * 0.1f;
}
static void FillDataF16(std::vector<uint16_t>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i)
        data[i] = FloatToFP16(((rand() % 2000) - 1000) * 0.1f);
}
static void FillDataBF16(std::vector<uint16_t>& data, int64_t N, int seed) {
    data.resize(N);
    srand(seed);
    for (int64_t i = 0; i < N; ++i)
        data[i] = FloatToBF16(((rand() % 2000) - 1000) * 0.1f);
}

static void FixDivisorI32(std::vector<int32_t>& y, int seed) {
    srand(seed);
    for (auto& v : y) if (v == 0) v = 1 + (rand() % 10);
}
static void FixDivisorU8(std::vector<uint8_t>& y, int seed) {
    srand(seed);
    for (auto& v : y) if (v == 0) v = 1 + (rand() % 10);
}
static void FixDivisorI8(std::vector<int8_t>& y, int seed) {
    srand(seed);
    for (auto& v : y) if (v == 0) v = (int8_t)(1 + (rand() % 10));
}
static void FixDivisorF32(std::vector<float>& y, int seed) {
    srand(seed);
    for (auto& v : y) if (std::fabs(v) < 0.01f) v = 1.0f;
}
static void FixDivisorF16(std::vector<uint16_t>& y, int seed) {
    srand(seed);
    for (auto& v : y) {
        float fv = FP16ToFloat(v);
        if (std::fabs(fv) < 0.01f) v = FloatToFP16(1.0f);
    }
}
static void FixDivisorBF16(std::vector<uint16_t>& y, int seed) {
    srand(seed);
    for (auto& v : y) {
        float fv = BF16ToFloat(v);
        if (std::fabs(fv) < 0.01f) v = FloatToBF16(1.0f);
    }
}

static int TestNegativeDivisorI32(aclrtStream stream) {
    printf("\n=== Test int32 negative divisor ===\n");
    int N = 50;
    std::vector<int32_t> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) { x[i] = (rand() % 200) - 100; y[i] = -1 - (rand() % 20); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT32, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i32);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesI32(aclrtStream stream) {
    printf("\n=== Test int32 negative values ===\n");
    int N = 50;
    std::vector<int32_t> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) { x[i] = -100 - (rand() % 100); y[i] = 1 + (rand() % 20); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT32, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i32);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestInt32Large(aclrtStream stream) {
    printf("\n=== Test int32 large values > 2^24 ===\n");
    int N = 50;
    std::vector<int32_t> x(N), y(N), out(N), expected;
    srand(44);
    for (int i = 0; i < N; ++i) { x[i] = ((rand() % 20000) + (1 << 24)); y[i] = 1 + (rand() % 1000); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT32, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i32);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroI32(aclrtStream stream) {
    printf("\n=== Test int32 divisor=0 ===\n");
    int N = 20;
    std::vector<int32_t> x(N), y(N, 0), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = (rand() % 200) - 100;
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT32, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i32);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

static int TestNegativeDivisorU8(aclrtStream stream) {
    printf("\n=== Test uint8 negative divisor ===\n");
    int N = 50;
    std::vector<uint8_t> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) { x[i] = (uint8_t)(rand() % 100); y[i] = (uint8_t)(200 + (rand() % 55)); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_UINT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_u8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesU8(aclrtStream stream) {
    printf("\n=== Test uint8 negative values ===\n");
    int N = 50;
    std::vector<uint8_t> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) { x[i] = (uint8_t)(rand() % 200); y[i] = (uint8_t)(1 + (rand() % 20)); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_UINT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_u8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroU8(aclrtStream stream) {
    printf("\n=== Test uint8 divisor=0 ===\n");
    int N = 20;
    std::vector<uint8_t> x(N), y(N, 0), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = (uint8_t)(rand() % 200);
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_UINT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_u8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

static int TestNegativeDivisorI8(aclrtStream stream) {
    printf("\n=== Test int8 negative divisor ===\n");
    int N = 50;
    std::vector<int8_t> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) { x[i] = (int8_t)((rand() % 100) - 50); y[i] = (int8_t)(-(rand() % 20) - 1); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesI8(aclrtStream stream) {
    printf("\n=== Test int8 negative values ===\n");
    int N = 50;
    std::vector<int8_t> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) { x[i] = (int8_t)(-(rand() % 100) - 1); y[i] = (int8_t)(1 + (rand() % 20)); }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroI8(aclrtStream stream) {
    printf("\n=== Test int8 divisor=0 ===\n");
    int N = 20;
    std::vector<int8_t> x(N), y(N, 0), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = (int8_t)((rand() % 200) - 100);
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_INT8, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_i8);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

static int TestMinimalF32(aclrtStream stream) {
    printf("\n=== Test float32 minimal: x=[5.5,7.2] y=[2,3] ===\n");
    std::vector<float> x = {5.5f, 7.2f};
    std::vector<float> y = {2.0f, 3.0f};
    std::vector<float> expected = {1.5f, 1.2f};
    std::vector<float> out;
    std::vector<int64_t> sh = {2};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT, stream)) return 1;
    for (size_t i = 0; i < out.size(); ++i)
        printf("  out[%zu] = %f\n", i, out[i]);
    int err = 0;
    if (!CmpResult(out, expected, 2, 1e-3f, 5e-2f, err)) { printf("  FAIL: %d errors\n", err); return 1; }
    printf("  PASS\n"); return 0;
}

static int TestNegativeDivisorF32(aclrtStream stream) {
    printf("\n=== Test float32 negative divisor ===\n");
    int N = 50;
    std::vector<float> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) { x[i] = ((rand() % 200) - 100) * 0.1f; y[i] = -(rand() % 20 + 1) * 0.1f; }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f32);
    int err = 0;
    if (!CmpResult(out, expected, N, 1e-3f, 5e-2f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesF32(aclrtStream stream) {
    printf("\n=== Test float32 negative values ===\n");
    int N = 50;
    std::vector<float> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) { x[i] = -(rand() % 100 + 1) * 0.1f; y[i] = (rand() % 20 + 1) * 0.1f; }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f32);
    int err = 0;
    if (!CmpResult(out, expected, N, 1e-3f, 5e-2f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroF32(aclrtStream stream) {
    printf("\n=== Test float32 divisor=0 ===\n");
    int N = 20;
    std::vector<float> x(N), y(N, 0.0f), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = ((rand() % 200) - 100) * 0.1f;
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f32);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

auto golden_f16_fn = [](uint16_t xh, uint16_t yh) -> uint16_t {
    float x = FP16ToFloat(xh), y = FP16ToFloat(yh);
    float r = golden_f32(x, y);
    return FloatToFP16(r);
};
auto golden_bf16_fn = [](uint16_t xh, uint16_t yh) -> uint16_t {
    float x = BF16ToFloat(xh), y = BF16ToFloat(yh);
    float r = golden_f32(x, y);
    return FloatToBF16(r);
};

static int TestNegativeDivisorF16(aclrtStream stream) {
    printf("\n=== Test float16 negative divisor ===\n");
    int N = 50;
    std::vector<uint16_t> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) {
        x[i] = FloatToFP16(((rand() % 200) - 100) * 0.1f);
        y[i] = FloatToFP16(-(rand() % 20 + 1) * 0.1f);
    }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f16_fn);
    int err = 0;
    if (!CmpResult(out, expected, N, 1e-3f, 5e-2f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesF16(aclrtStream stream) {
    printf("\n=== Test float16 negative values ===\n");
    int N = 50;
    std::vector<uint16_t> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) {
        x[i] = FloatToFP16(-(rand() % 100 + 1) * 0.1f);
        y[i] = FloatToFP16((rand() % 20 + 1) * 0.1f);
    }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f16_fn);
    int err = 0;
    if (!CmpResult(out, expected, N, 1e-3f, 5e-2f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroF16(aclrtStream stream) {
    printf("\n=== Test float16 divisor=0 ===\n");
    int N = 20;
    std::vector<uint16_t> x(N), y(N, FloatToFP16(0.0f)), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = FloatToFP16(((rand() % 200) - 100) * 0.1f);
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_FLOAT16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_f16_fn);
    int err = 0;
    if (!CmpResult(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

static int TestNegativeDivisorBF16(aclrtStream stream) {
    printf("\n=== Test bfloat16 negative divisor ===\n");
    int N = 50;
    std::vector<uint16_t> x(N), y(N), out(N), expected;
    srand(42);
    for (int i = 0; i < N; ++i) {
        x[i] = FloatToBF16(((rand() % 200) - 100) * 0.1f);
        y[i] = FloatToBF16(-(rand() % 20 + 1) * 0.1f);
    }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_BF16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_bf16_fn);
    int err = 0;
    if (!CmpResultBF16(out, expected, N, 5e-2f, 5e-1f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestNegativeValuesBF16(aclrtStream stream) {
    printf("\n=== Test bfloat16 negative values ===\n");
    int N = 50;
    std::vector<uint16_t> x(N), y(N), out(N), expected;
    srand(43);
    for (int i = 0; i < N; ++i) {
        x[i] = FloatToBF16(-(rand() % 100 + 1) * 0.1f);
        y[i] = FloatToBF16((rand() % 20 + 1) * 0.1f);
    }
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_BF16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_bf16_fn);
    int err = 0;
    if (!CmpResultBF16(out, expected, N, 5e-2f, 5e-1f, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}
static int TestDivisorZeroBF16(aclrtStream stream) {
    printf("\n=== Test bfloat16 divisor=0 ===\n");
    int N = 20;
    std::vector<uint16_t> x(N), y(N, FloatToBF16(0.0f)), out(N), expected;
    srand(45);
    for (int i = 0; i < N; ++i) x[i] = FloatToBF16(((rand() % 200) - 100) * 0.1f);
    std::vector<int64_t> sh = {N};
    if (RunOp(x, sh, y, sh, out, sh, ACL_BF16, stream)) return 1;
    ComputeGoldenBroadcast(x, sh, y, sh, expected, sh, golden_bf16_fn);
    int err = 0;
    if (!CmpResultBF16(out, expected, N, 0, 0, err)) { printf("  FAIL: %d/%d errors\n", err, N); return 1; }
    printf("  PASS\n"); return 0;
}

static int RunAllTestsI32(aclrtStream stream) {
    printf("\n========== Test int32 (ACL_INT32) ==========\n");
    int fail = 0;
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<int32_t> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataI32(x, nx, seed); FillDataI32(y, ny, seed + 100); FixDivisorI32(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_INT32, stream)) {
            printf("  [int32][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_i32);
        int err = 0;
        if (!CmpResult(out, expected, no, 0, 0, err)) { printf("  [int32][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [int32][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorI32(stream);
    fail += TestNegativeValuesI32(stream);
    fail += TestInt32Large(stream);
    fail += TestDivisorZeroI32(stream);
    return fail;
}

static int RunAllTestsU8(aclrtStream stream) {
    printf("\n========== Test uint8 (ACL_UINT8) ==========\n");
    int fail = 0;
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<uint8_t> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataU8(x, nx, seed); FillDataU8(y, ny, seed + 100); FixDivisorU8(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_UINT8, stream)) {
            printf("  [uint8][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_u8);
        int err = 0;
        if (!CmpResult(out, expected, no, 0, 0, err)) { printf("  [uint8][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [uint8][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorU8(stream);
    fail += TestNegativeValuesU8(stream);
    fail += TestDivisorZeroU8(stream);
    return fail;
}

static int RunAllTestsI8(aclrtStream stream) {
    printf("\n========== Test int8 (ACL_INT8) ==========\n");
    int fail = 0;
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<int8_t> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataI8(x, nx, seed); FillDataI8(y, ny, seed + 100); FixDivisorI8(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_INT8, stream)) {
            printf("  [int8][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_i8);
        int err = 0;
        if (!CmpResult(out, expected, no, 0, 0, err)) { printf("  [int8][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [int8][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorI8(stream);
    fail += TestNegativeValuesI8(stream);
    fail += TestDivisorZeroI8(stream);
    return fail;
}

static int RunAllTestsF32(aclrtStream stream) {
    printf("\n========== Test float32 (ACL_FLOAT) ==========\n");
    int fail = 0;
    fail += TestMinimalF32(stream);
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<float> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataF32(x, nx, seed); FillDataF32(y, ny, seed + 100); FixDivisorF32(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_FLOAT, stream)) {
            printf("  [float32][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_f32);
        int err = 0;
        if (!CmpResult(out, expected, no, 1e-3f, 5e-2f, err)) { printf("  [float32][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [float32][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorF32(stream);
    fail += TestNegativeValuesF32(stream);
    fail += TestDivisorZeroF32(stream);
    return fail;
}

static int RunAllTestsF16(aclrtStream stream) {
    printf("\n========== Test float16 (ACL_FLOAT16) ==========\n");
    int fail = 0;
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<uint16_t> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataF16(x, nx, seed); FillDataF16(y, ny, seed + 100); FixDivisorF16(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_FLOAT16, stream)) {
            printf("  [float16][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_f16_fn);
        int err = 0;
        if (!CmpResult(out, expected, no, 1e-3f, 5e-2f, err)) { printf("  [float16][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [float16][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorF16(stream);
    fail += TestNegativeValuesF16(stream);
    fail += TestDivisorZeroF16(stream);
    return fail;
}

static int RunAllTestsBF16(aclrtStream stream) {
    printf("\n========== Test bfloat16 (ACL_BF16) ==========\n");
    int fail = 0;
    auto cases = GetTestCases();
    for (auto& tc : cases) {
        int64_t nx = ShapeSize(tc.xShape), ny = ShapeSize(tc.yShape);
        auto outShape = BroadcastShape(tc.xShape, tc.yShape);
        if (outShape.empty()) { printf("  [%s] skip (incompatible shapes)\n", tc.name.c_str()); continue; }
        int64_t no = ShapeSize(outShape);
        if (no == 0) { printf("  [%s] skip (empty output)\n", tc.name.c_str()); continue; }
        std::vector<uint16_t> x, y, out, expected;
        int seed = 100 + fail * 7;
        FillDataBF16(x, nx, seed); FillDataBF16(y, ny, seed + 100); FixDivisorBF16(y, seed + 200);
        if (RunOp(x, tc.xShape, y, tc.yShape, out, outShape, ACL_BF16, stream)) {
            printf("  [bfloat16][%s] RUN FAIL\n", tc.name.c_str()); fail++; continue;
        }
        ComputeGoldenBroadcast(x, tc.xShape, y, tc.yShape, expected, outShape, golden_bf16_fn);
        int err = 0;
        if (!CmpResultBF16(out, expected, no, 5e-2f, 5e-1f, err)) { printf("  [bfloat16][%s] %d/%ld errors -> FAIL\n", tc.name.c_str(), err, no); fail++; }
        else printf("  [bfloat16][%s] PASS\n", tc.name.c_str());
    }
    fail += TestNegativeDivisorBF16(stream);
    fail += TestNegativeValuesBF16(stream);
    fail += TestDivisorZeroBF16(stream);
    return fail;
}

int main()
{
    auto r = aclInit(nullptr); CHECK_RET(r == ACL_SUCCESS, "aclInit");
    r = aclrtSetDevice(0); CHECK_RET(r == ACL_SUCCESS, "setDevice");
    aclrtStream stream = nullptr;
    r = aclrtCreateStream(&stream); CHECK_RET(r == ACL_SUCCESS, "createStream");

    int fail = 0;
    fail += RunAllTestsI32(stream);
    fail += RunAllTestsU8(stream);
    fail += RunAllTestsI8(stream);
    fail += RunAllTestsF32(stream);
    fail += RunAllTestsF16(stream);
    fail += RunAllTestsBF16(stream);

    printf("\n========================================\n");
    if (fail) printf("RESULT: SOME TESTS FAILED (fail=%d)\n", fail);
    else printf("RESULT: ALL TESTS PASSED\n");
    printf("========================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return fail ? 1 : 0;
}
