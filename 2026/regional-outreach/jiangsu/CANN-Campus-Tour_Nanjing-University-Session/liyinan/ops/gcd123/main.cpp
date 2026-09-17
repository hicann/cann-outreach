/**
 * @file main.cpp
 * Host launcher for gcd123 (CPU / NPU).
 * Inputs are already broadcast to out shape [1, 4, 16, 128].
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void gcd123_do(uint32_t blockDim, void *stream, uint8_t *self, uint8_t *other, uint8_t *out);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void gcd123(GM_ADDR self, GM_ADDR other, GM_ADDR out);
#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    // broadcasted out shape [1, 4, 16, 128]
    size_t byteSize = 1 * 4 * 16 * 128 * sizeof(uint16_t);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *self = (uint8_t *)AscendC::GmAlloc(byteSize);
    uint8_t *other = (uint8_t *)AscendC::GmAlloc(byteSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(byteSize);
    ReadFile("./input/input_self.bin", byteSize, self, byteSize);
    ReadFile("./input/input_other.bin", byteSize, other, byteSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(gcd123, blockDim, self, other, out);
    WriteFile("./output/output_out.bin", out, byteSize);
    AscendC::GmFree((void *)self);
    AscendC::GmFree((void *)other);
    AscendC::GmFree((void *)out);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *selfHost, *otherHost, *outHost;
    uint8_t *selfDevice, *otherDevice, *outDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&selfHost), byteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&otherHost), byteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&outHost), byteSize));
    CHECK_ACL(aclrtMalloc((void **)&selfDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&otherDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_self.bin", byteSize, selfHost, byteSize);
    ReadFile("./input/input_other.bin", byteSize, otherHost, byteSize);
    CHECK_ACL(aclrtMemcpy(selfDevice, byteSize, selfHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(otherDevice, byteSize, otherHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    gcd123_do(blockDim, stream, selfDevice, otherDevice, outDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_out.bin", outHost, byteSize);

    CHECK_ACL(aclrtFree(selfDevice));
    CHECK_ACL(aclrtFree(otherDevice));
    CHECK_ACL(aclrtFree(outDevice));
    CHECK_ACL(aclrtFreeHost(selfHost));
    CHECK_ACL(aclrtFreeHost(otherHost));
    CHECK_ACL(aclrtFreeHost(outHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
