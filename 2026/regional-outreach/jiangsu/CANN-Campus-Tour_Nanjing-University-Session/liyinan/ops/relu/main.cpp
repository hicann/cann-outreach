/**
 * @file main.cpp
 * Host launcher for relu_custom (CPU / NPU).
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void relu_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y);
#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t byteSize = 2 * 2 * 4 * 128 * sizeof(uint16_t);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(byteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(byteSize);
    ReadFile("./input/input_x.bin", byteSize, x, byteSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(relu_custom, blockDim, x, y);
    WriteFile("./output/output_y.bin", y, byteSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *xDevice, *yDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&xHost), byteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), byteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", byteSize, xHost, byteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, byteSize, xHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    relu_custom_do(blockDim, stream, xDevice, yDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(yHost, byteSize, yDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", yHost, byteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
