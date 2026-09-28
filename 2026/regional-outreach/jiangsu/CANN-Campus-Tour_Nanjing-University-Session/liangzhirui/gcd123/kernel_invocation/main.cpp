/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 直接调用（kernel_invocation）模式的 host 侧测试程序。
 * 支持两种运行方式：
 *   - NPU 模式：通过 AscendCL 在真实昇腾硬件上执行；
 *   - CPU 模式（CPU 仿真调试）：通过 tikicpulib 在 x86 上仿真执行，无需 NPU 硬件。
 *
 * 数据准备：运行 gen_data_and_tiling.py 生成 input/self.bin、input/other.bin、
 *           input/tiling.bin、output/golden.bin。
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "kernel_invocation/gcd123_tiling_def.h"

#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void gcd123_custom(GM_ADDR self, GM_ADDR other, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling);
#else
#include "acl/acl.h"
extern void gcd123_custom_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* self,
    uint8_t* other, uint8_t* out, uint8_t* workspace, uint8_t* tiling);
#endif

#define CHECK_ACL(x)                                                                      \
    do {                                                                                  \
        aclError ret = (x);                                                               \
        if (ret != ACL_SUCCESS) {                                                         \
            printf("ACL error: %s:%d, ret=%d\n", __FILE__, __LINE__, (int)ret);           \
            return -1;                                                                    \
        }                                                                                 \
    } while (0)

static int32_t ReadFile(const char* path, size_t size, uint8_t* buf, size_t bufSize)
{
    if (size > bufSize) {
        printf("read buffer too small: %s\n", path);
        return -1;
    }
    FILE* fp = fopen(path, "rb");
    if (fp == nullptr) {
        printf("open file failed: %s\n", path);
        return -1;
    }
    size_t ret = fread(buf, 1, size, fp);
    fclose(fp);
    if (ret != size) {
        printf("read file failed: %s\n", path);
        return -1;
    }
    return 0;
}

static int32_t WriteFile(const char* path, uint8_t* buf, size_t size)
{
    FILE* fp = fopen(path, "wb");
    if (fp == nullptr) {
        printf("create file failed: %s\n", path);
        return -1;
    }
    size_t ret = fwrite(buf, 1, size, fp);
    fclose(fp);
    if (ret != size) {
        printf("write file failed: %s\n", path);
        return -1;
    }
    return 0;
}

// host 侧 fp16 -> float 转换（CPU/NPU 模式通用，不依赖 acl）
static inline float HalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; // +-0
        } else {
            // 非规格化数：归一化
            int32_t e = -14;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                e--;
            }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13); // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// 与 output/golden.bin 比对（整数域输入，允许 0.5 容差）
static int32_t VerifyResult(const uint8_t* outBuf, size_t outByteSize)
{
    size_t elemNum = outByteSize / sizeof(uint16_t);
    uint8_t* goldenBuf = new (std::nothrow) uint8_t[outByteSize];
    if (goldenBuf == nullptr) {
        printf("alloc golden buffer failed\n");
        return -1;
    }
    if (ReadFile("./output/golden.bin", outByteSize, goldenBuf, outByteSize) != 0) {
        delete[] goldenBuf;
        return -1;
    }
    const uint16_t* outHalf = reinterpret_cast<const uint16_t*>(outBuf);
    const uint16_t* goldHalf = reinterpret_cast<const uint16_t*>(goldenBuf);
    size_t bad = 0;
    double maxDiff = 0.0;
    for (size_t i = 0; i < elemNum; i++) {
        double a = static_cast<double>(HalfToFloat(outHalf[i]));
        double b = static_cast<double>(HalfToFloat(goldHalf[i]));
        double diff = std::fabs(a - b);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        if (diff > 0.5) {
            bad++;
        }
    }
    delete[] goldenBuf;
    if (bad == 0) {
        printf("test pass: %zu elements all match golden (max diff=%f)\n", elemNum, maxDiff);
        return 0;
    }
    printf("test failed: %zu / %zu elements mismatch (max diff=%f)\n", bad, elemNum, maxDiff);
    return -1;
}

int32_t main(int32_t argc, char* argv[])
{
    size_t tilingSize = sizeof(Gcd123TilingData);
    const size_t usrWorkspaceSize = 4096;
    const size_t sysWorkspaceSize = 16 * 1024 * 1024;
    const size_t workspaceByteSize = usrWorkspaceSize + sysWorkspaceSize;

#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
    // ---------- CPU 仿真调试模式 ----------
    AscendC::SetKernelMode(AscendC::KernelMode::AIV_MODE);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* usrWorkSpace = (uint8_t*)AscendC::GmAlloc(workspaceByteSize);
    if (ReadFile("./input/tiling.bin", tilingSize, tiling, tilingSize) != 0) {
        return -1;
    }
    const Gcd123TilingData* tilingHost = reinterpret_cast<const Gcd123TilingData*>(tiling);
    uint32_t blockDim = tilingHost->blockDim;
    size_t inputByteSize = (size_t)tilingHost->totalLength * sizeof(uint16_t);
    size_t outputByteSize = inputByteSize;

    uint8_t* self = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* other = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    if (ReadFile("./input/self.bin", inputByteSize, self, inputByteSize) != 0 ||
        ReadFile("./input/other.bin", inputByteSize, other, inputByteSize) != 0) {
        return -1;
    }
    ICPU_RUN_KF(gcd123_custom, blockDim, self, other, out, usrWorkSpace, tiling);
    if (WriteFile("./output/out.bin", out, outputByteSize) != 0) {
        return -1;
    }
    int32_t verifyRet = VerifyResult(out, outputByteSize);
    AscendC::GmFree((void*)self);
    AscendC::GmFree((void*)other);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)usrWorkSpace);
    AscendC::GmFree((void*)tiling);
    return verifyRet;
#else
    // ---------- NPU 真机模式 ----------
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *selfHost, *otherHost, *outHost, *tilingHost, *workspaceHost;
    uint8_t *selfDevice, *otherDevice, *outDevice, *tilingDevice, *workspaceDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingSize));
    if (ReadFile("./input/tiling.bin", tilingSize, tilingHost, tilingSize) != 0) {
        return -1;
    }
    uint32_t blockDim = reinterpret_cast<const Gcd123TilingData*>(tilingHost)->blockDim;
    size_t inputByteSize =
        (size_t)reinterpret_cast<const Gcd123TilingData*>(tilingHost)->totalLength * sizeof(uint16_t);
    size_t outputByteSize = inputByteSize;

    CHECK_ACL(aclrtMallocHost((void**)(&selfHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&otherHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&outHost), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&workspaceHost), workspaceByteSize));
    CHECK_ACL(aclrtMalloc((void**)&selfDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&otherDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&outDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    if (ReadFile("./input/self.bin", inputByteSize, selfHost, inputByteSize) != 0 ||
        ReadFile("./input/other.bin", inputByteSize, otherHost, inputByteSize) != 0) {
        return -1;
    }
    CHECK_ACL(aclrtMemcpy(selfDevice, inputByteSize, selfHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(otherDevice, inputByteSize, otherHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    gcd123_custom_do(blockDim, nullptr, stream, selfDevice, otherDevice, outDevice,
        workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(outHost, outputByteSize, outDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (WriteFile("./output/out.bin", outHost, outputByteSize) != 0) {
        return -1;
    }
    int32_t verifyRet = VerifyResult(outHost, outputByteSize);

    CHECK_ACL(aclrtFree(selfDevice));
    CHECK_ACL(aclrtFree(otherDevice));
    CHECK_ACL(aclrtFree(outDevice));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(selfHost));
    CHECK_ACL(aclrtFreeHost(otherHost));
    CHECK_ACL(aclrtFreeHost(outHost));
    CHECK_ACL(aclrtFreeHost(workspaceHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return verifyRet;
#endif
}
