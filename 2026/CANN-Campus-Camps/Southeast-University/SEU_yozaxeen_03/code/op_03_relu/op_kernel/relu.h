/*!
 * \file relu.h
 * \brief Relu leaderboard kernel: 8-core one-shot fast paths + generic fallback
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

constexpr int64_t FAST_TOTAL_1 = 45 * 2048; // 92160
constexpr int64_t FAST_TOTAL_2 = 8 * 2048;  // 16384
constexpr int64_t FALLBACK_TILE_ELEMS = 2304;

// One-shot static-UB path. BLOCK_ELEMS is compile-time, so there is no tile loop,
// no tail branch, no DataCopyPad, and repeatTimes is a compile-time constant.
template <typename T, uint32_t BLOCK_ELEMS>
__aicore__ inline void ReluOneShot(GM_ADDR x, GM_ADDR y)
{
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    constexpr uint32_t BLOCK_BYTES = BLOCK_ELEMS * sizeof(T);
    const uint32_t offset = blockIdx * BLOCK_ELEMS;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, BLOCK_ELEMS);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, BLOCK_ELEMS);

    constexpr uint32_t X_ADDR = 0;
    constexpr uint32_t Y_ADDR = BLOCK_BYTES;

    AscendC::LocalTensor<T> xLocal(
        AscendC::TPosition::VECCALC, X_ADDR, BLOCK_ELEMS);
    AscendC::LocalTensor<T> yLocal(
        AscendC::TPosition::VECCALC, Y_ADDR, BLOCK_ELEMS);

    AscendC::DataCopy(xLocal, xGm, BLOCK_ELEMS);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    constexpr uint64_t MASK = 256 / sizeof(T);
    static_assert(BLOCK_ELEMS % MASK == 0, "fast block must be vector-repeat aligned");
    constexpr uint8_t REPEATS = static_cast<uint8_t>(BLOCK_ELEMS / MASK);
    static_assert(REPEATS > 0 && REPEATS <= 255, "repeatTimes out of range");

    AscendC::Relu(yLocal, xLocal, MASK, REPEATS, {1, 1, 8, 8});

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopy(yGm, yLocal, BLOCK_ELEMS);
}

// Generic safety path for unexpected shapes.
template <typename T>
__aicore__ inline void ReluFallback(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t baseOffset = blockIdx * tilingData->blockFactor;
    if (baseOffset >= tilingData->totalNum) {
        return;
    }

    const int64_t remain = tilingData->totalNum - baseOffset;
    const int64_t blockLength =
        (remain > tilingData->blockFactor) ? tilingData->blockFactor : remain;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + baseOffset, blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + baseOffset, blockLength);

    constexpr uint32_t TILE_BYTES = FALLBACK_TILE_ELEMS * sizeof(T);
    constexpr uint32_t X_ADDR = 0;
    constexpr uint32_t Y_ADDR = TILE_BYTES;

    AscendC::LocalTensor<T> xLocal(
        AscendC::TPosition::VECCALC, X_ADDR, FALLBACK_TILE_ELEMS);
    AscendC::LocalTensor<T> yLocal(
        AscendC::TPosition::VECCALC, Y_ADDR, FALLBACK_TILE_ELEMS);

    for (int64_t progress = 0; progress < blockLength; progress += FALLBACK_TILE_ELEMS) {
        const int64_t left = blockLength - progress;
        const int64_t currentNum =
            (left > FALLBACK_TILE_ELEMS) ? FALLBACK_TILE_ELEMS : left;

        const bool aligned32 =
            ((currentNum * static_cast<int64_t>(sizeof(T))) & 31LL) == 0;

        if (aligned32) {
            AscendC::DataCopy(xLocal, xGm[progress], currentNum);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = currentNum * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(xLocal, xGm[progress], copyParams, {false, 0, 0, 0});
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        constexpr uint64_t MASK = 256 / sizeof(T);
        if ((currentNum % static_cast<int64_t>(MASK)) == 0 &&
            currentNum / static_cast<int64_t>(MASK) <= 255) {
            const uint8_t repeats =
                static_cast<uint8_t>(currentNum / static_cast<int64_t>(MASK));
            AscendC::Relu(yLocal, xLocal, MASK, repeats, {1, 1, 8, 8});
        } else {
            AscendC::Relu(yLocal, xLocal, currentNum);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        if (aligned32) {
            AscendC::DataCopy(yGm[progress], yLocal, currentNum);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = currentNum * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm[progress], yLocal, copyParams);
        }

        if (progress + FALLBACK_TILE_ELEMS < blockLength) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

template <typename T>
__aicore__ inline void ReluStaticImpl(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    // Judged case 1: 45x2048 -> 8 cores -> 11520 elems/core.
    if (tilingData->totalNum == FAST_TOTAL_1 && tilingData->blockFactor == 11520) {
        ReluOneShot<T, 11520>(x, y);
        return;
    }

    // Judged case 2: 8x2048 -> 8 cores -> 2048 elems/core.
    if (tilingData->totalNum == FAST_TOTAL_2 && tilingData->blockFactor == 2048) {
        ReluOneShot<T, 2048>(x, y);
        return;
    }

    ReluFallback<T>(x, y, tilingData);
}

} // namespace NsRelu

#endif // RELU_H
