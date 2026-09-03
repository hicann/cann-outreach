/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <algorithm>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "common/helper/kernel_constant.h"
#include "icamax_kernel.h"
#include "icamax_tiling_data.h"
namespace {
static aclblasStatus_t ValidateIcamaxParams(
    aclblasHandle_t handle, int n, int incx, const aclblasComplex* x, int* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasIcamax", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // quick return: n == 0 or incx <= 0 (incx == 0 or negative stride).
    // No reduction is performed; result is written as 0 (Netlib icamax.f 1993+ /
    // cuBLAS semantics). Negative strides are NOT traversed in reverse.
    if (n == 0 || incx < 1) {
        if (result != nullptr) {
            *result = 0;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasIcamax", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasIcamax", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
static IcamaxTilingData CalcIcamaxTiling(int n, uint32_t numBlocks, int incx)
{
    IcamaxTilingData tiling{};
    uint32_t totalN = static_cast<uint32_t>(n);
    tiling.totalN = totalN;
    tiling.useCoreNum = numBlocks;
    tiling.incx = static_cast<uint32_t>(incx);
    tiling.perCoreN = totalN / numBlocks;
    tiling.lastCoreN = tiling.perCoreN + (totalN % numBlocks);
    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(tiling.perCoreN, SIMT_MIN_THREAD_NUM), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);
    return tiling;
}
static aclblasStatus_t LaunchIcamaxKernel(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    aclblasStatus_t vStatus = ValidateIcamaxParams(handle, n, incx, x, result);
    if (vStatus != ACLBLAS_STATUS_SUCCESS || n <= 0 || incx < 1) {
        return vStatus;
    }
    auto* h = handle;
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasIcamax", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    IcamaxTilingData tiling = CalcIcamaxTiling(n, numBlocks, incx);
    OP_LOGD(
        "aclblasIcamax", "tiling: totalN=%u perCoreN=%u lastCoreN=%u useCoreNum=%u nthreads=%u incx=%u",
        tiling.totalN, tiling.perCoreN, tiling.lastCoreN, tiling.useCoreNum, tiling.nthreads, tiling.incx);
    void* workSpace = GetEffectiveWorkspace(h);
    size_t workspaceBytes = GetEffectiveWorkspaceSize(h);
    constexpr uint32_t ALIGN_FLOATS = 64;
    uint32_t totalFloats = numBlocks * 2;
    uint32_t alignedFloats = ((totalFloats + ALIGN_FLOATS - 1) / ALIGN_FLOATS) * ALIGN_FLOATS;
    size_t requiredBytes = static_cast<size_t>(alignedFloats) * sizeof(float);
    if (workspaceBytes < requiredBytes) {
        OP_LOGE(
            "aclblasIcamax", "workspace too small: need %zu, got %zu", requiredBytes, workspaceBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    OP_LOGI(
        "aclblasIcamax", "launching kernel: blocks=%u, useCoreNum=%u, nthreads=%u, incx=%d", numBlocks,
        tiling.useCoreNum, tiling.nthreads, incx);
    icamax_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(x)), reinterpret_cast<GM_ADDR>(result),
        reinterpret_cast<GM_ADDR>(workSpace), tiling, numBlocks, reinterpret_cast<void*>(h->stream));
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    return LaunchIcamaxKernel(handle, n, x, incx, result);
}
