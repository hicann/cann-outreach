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
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "icamax_tiling_data.h"
using namespace AscendC;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32;
constexpr uint32_t COMPLEX_FLOAT_PER_ELEM = 2; // each complex64 element = 2 x float32 (real, imag)

// ─────────────────────────────────────────────────────────────────────────────
// SIMT block reduction over complex64 elements.
//
// "Modulus" for icamax follows Netlib SCABS1 (1-norm): |x[k]| = |Re| + |Im|,
// NOT the Euclidean norm sqrt(Re^2 + Im^2). Comparison uses "strictly greater
// than" so that ties keep the smallest index and NaN never wins; the first +Inf
// element wins (aligned with Netlib icamax / cuBLAS semantics).
// ─────────────────────────────────────────────────────────────────────────────
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcamaxSimtCompute(
    uint32_t calNum, uint32_t blockStartOffset, uint32_t stride, __gm__ const float* xGm, __gm__ float* wsSlotPtr)
{
    __ubuf__ float ubPartialVals[SIMT_MAX_THREAD_NUM];
    __ubuf__ uint32_t ubPartialIdxs[SIMT_MAX_THREAD_NUM];
    float bestVal = 0.0f;
    uint32_t bestIdx = 0;
    bool hasValue = false;
    if (calNum > 0 && threadIdx.x < calNum) {
        for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
            uint32_t elemIdx = blockStartOffset + i;
            uint32_t floatBase = elemIdx * stride * COMPLEX_FLOAT_PER_ELEM;
            float reVal = xGm[floatBase];
            float imVal = xGm[floatBase + 1];
            float absRe = (reVal >= 0.0f) ? reVal : -reVal;
            float absIm = (imVal >= 0.0f) ? imVal : -imVal;
            float modVal = absRe + absIm;
            if (!(modVal != modVal) && (!hasValue || modVal > bestVal || (modVal == bestVal && elemIdx < bestIdx))) {
                bestVal = modVal;
                bestIdx = elemIdx;
                hasValue = true;
            }
        }
    }
    ubPartialVals[threadIdx.x] = bestVal;
    ubPartialIdxs[threadIdx.x] = bestIdx;
    asc_syncthreads();
    unsigned int blockPow2 = 1;
    while (blockPow2 < blockDim.x)
        blockPow2 <<= 1;
    for (unsigned int s = blockPow2 >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
            float otherVal = ubPartialVals[threadIdx.x + s];
            uint32_t otherIdx = ubPartialIdxs[threadIdx.x + s];
            if (!(otherVal != otherVal) && (otherVal > ubPartialVals[threadIdx.x] ||
                (otherVal == ubPartialVals[threadIdx.x] && otherIdx < ubPartialIdxs[threadIdx.x]))) {
                ubPartialVals[threadIdx.x] = otherVal;
                ubPartialIdxs[threadIdx.x] = otherIdx;
            }
        }
        asc_syncthreads();
    }
    if (threadIdx.x == 0) {
        wsSlotPtr[0] = ubPartialVals[0];
        reinterpret_cast<__gm__ uint32_t*>(wsSlotPtr + 1)[0] = ubPartialIdxs[0];
    }
}

extern "C" __global__ __aicore__ void icamax_simt_kernel(GM_ADDR x, GM_ADDR workSpace, IcamaxTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int32_t blockIdx = GetBlockIdx();
    uint32_t blockId = static_cast<uint32_t>(blockIdx);
    uint32_t calNum = (blockId == tdata.useCoreNum - 1) ? tdata.lastCoreN : tdata.perCoreN;
    uint32_t startOffset = blockId * tdata.perCoreN;
    if (calNum > 0) {
        uint32_t wsBase = blockId * 2;
        __gm__ float* wsSlotPtr = reinterpret_cast<__gm__ float*>(workSpace) + wsBase;
        asc_vf_call<IcamaxSimtCompute>(dim3{tdata.nthreads, 1, 1}, calNum, startOffset, tdata.incx,
            reinterpret_cast<__gm__ const float*>(x), wsSlotPtr);
    }
}

extern "C" __global__ __aicore__ void icamax_reduce_kernel(GM_ADDR workSpace, GM_ADDR resultGM, IcamaxTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    uint32_t useCoreNum = tdata.useCoreNum;
    uint32_t totalFloats = useCoreNum * 2;
    constexpr uint32_t ALIGN_FLOATS = ELEMENTS_PER_REPEAT;
    uint32_t alignedFloats = ((totalFloats + ALIGN_FLOATS - 1) / ALIGN_FLOATS) * ALIGN_FLOATS;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, alignedFloats * sizeof(float));
    LocalTensor<float> wsData = buf.Get<float>();
    GlobalTensor<float> wsGM;
    wsGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), alignedFloats);
    DataCopy(wsData, wsGM, alignedFloats);
    float bestVal = 0.0f;
    uint32_t bestIdx = 0;
    bool hasValue = false;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        float val = wsData.GetValue(i * 2);
        float idxFloat = wsData.GetValue(i * 2 + 1);
        uint32_t idx = *reinterpret_cast<uint32_t*>(&idxFloat);
        if (!(val != val) && (!hasValue || val > bestVal || (val == bestVal && idx < bestIdx))) {
            bestVal = val;
            bestIdx = idx;
            hasValue = true;
        }
    }
    if (!hasValue) {
        bestVal = 0.0f;
        bestIdx = 0;
    }
    int32_t result = static_cast<int32_t>(bestIdx) + 1;
    TBuf<TPosition::VECCALC> outBuf;
    pipe.InitBuffer(outBuf, UB_BYTENUM_PER_BLOCK);
    LocalTensor<int32_t> outLocal = outBuf.Get<int32_t>();
    Duplicate<int32_t>(outLocal, result, ELEMENTS_PER_BLOCK);
    GlobalTensor<int32_t> outGM;
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(resultGM), 1);
    DataCopyParams writeParams{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(outGM, outLocal, writeParams);
}

void icamax_kernel_do(
    GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, const IcamaxTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    icamax_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, workSpace, tiling);
    icamax_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, result, tiling);
}
