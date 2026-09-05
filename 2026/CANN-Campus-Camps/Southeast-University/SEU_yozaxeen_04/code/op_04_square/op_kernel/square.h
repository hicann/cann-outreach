/*!
 * \file square.h
 * \brief Square leaderboard-extreme kernel.
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

constexpr uint32_t MAX_VECTOR_REPEATS = 255;
constexpr uint32_t MAX_TILE_BYTES = MAX_VECTOR_REPEATS * 256; // 65280B per tensor
constexpr uint32_t FLAG_UNIFORM_BLOCKS = 1U << 0;
constexpr uint32_t FLAG_LAST_COPY_ALIGNED = 1U << 1;

template <typename T>
__aicore__ inline void MulFullRepeats(
    AscendC::LocalTensor<T>& dst,
    AscendC::LocalTensor<T>& src,
    uint32_t repeats)
{
    if (repeats != 0U) {
        constexpr uint32_t ELEMS_PER_REPEAT = 256 / sizeof(T);
        AscendC::Mul(
            dst,
            src,
            src,
            static_cast<uint64_t>(ELEMS_PER_REPEAT),
            static_cast<uint8_t>(repeats),
            {1, 1, 1, 8, 8, 8});
    }
}

template <typename T>
__aicore__ inline void MulTail(
    AscendC::LocalTensor<T>& dst,
    AscendC::LocalTensor<T>& src,
    uint32_t fullRepeats,
    uint32_t tail)
{
    constexpr uint32_t ELEMS_PER_REPEAT = 256 / sizeof(T);

    if (fullRepeats != 0U) {
        AscendC::Mul(
            dst,
            src,
            src,
            static_cast<uint64_t>(ELEMS_PER_REPEAT),
            static_cast<uint8_t>(fullRepeats),
            {1, 1, 1, 8, 8, 8});
    }

    if (tail != 0U) {
        const uint32_t offset = fullRepeats * ELEMS_PER_REPEAT;
        AscendC::Mul(
            dst[offset],
            src[offset],
            src[offset],
            static_cast<uint64_t>(tail),
            static_cast<uint8_t>(1),
            {1, 1, 1, 8, 8, 8});
    }
}

template <typename T>
__aicore__ inline void SquareExtremeImpl(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t blockFactor = tilingData->blockFactor;
    const uint32_t baseOffset = blockIdx * blockFactor;

    constexpr uint32_t MAX_TILE_ELEMS = MAX_TILE_BYTES / sizeof(T);
    constexpr uint32_t X_ADDR = 0;
    constexpr uint32_t Y_ADDR = MAX_TILE_BYTES;

    AscendC::LocalTensor<T> xLocal(
        AscendC::TPosition::VECCALC, X_ADDR, MAX_TILE_ELEMS);
    AscendC::LocalTensor<T> yLocal(
        AscendC::TPosition::VECCALC, Y_ADDR, MAX_TILE_ELEMS);

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    // ------------------------------------------------------------
    // Uniform fast path: all cores have exactly the same 256B-aligned
    // block.  This is intentionally kept almost identical to the proven
    // Sub/Mul static-UB leaderboard kernel: one copy, one vector program,
    // one copy, and no tile/padding/tail bookkeeping.
    // ------------------------------------------------------------
    if ((tilingData->flags & FLAG_UNIFORM_BLOCKS) != 0U) {
        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input_x) + baseOffset,
            blockFactor);
        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output) + baseOffset,
            blockFactor);

        AscendC::DataCopy(xLocal, xGm, blockFactor);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        MulFullRepeats<T>(yLocal, xLocal, tilingData->normalRepeats);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopy(yGm, yLocal, blockFactor);
        return;
    }

    // ------------------------------------------------------------
    // Non-uniform case.  Every core except the final one still runs the
    // same zero-tail fast path.  Only one core pays DataCopyPad / vector
    // tail overhead.
    // ------------------------------------------------------------
    const bool isLast = (blockIdx + 1U == tilingData->blockNum);

    if (!isLast) {
        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input_x) + baseOffset,
            blockFactor);
        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output) + baseOffset,
            blockFactor);

        AscendC::DataCopy(xLocal, xGm, blockFactor);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        MulFullRepeats<T>(yLocal, xLocal, tilingData->normalRepeats);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopy(yGm, yLocal, blockFactor);
        return;
    }

    const uint32_t count = tilingData->lastBlockLength;

    xGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(input_x) + baseOffset,
        count);
    yGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(output) + baseOffset,
        count);

    if ((tilingData->flags & FLAG_LAST_COPY_ALIGNED) != 0U) {
        AscendC::DataCopy(xLocal, xGm, count);
    } else {
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = count * sizeof(T);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm, p, {false, 0, 0, 0});
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    MulTail<T>(
        yLocal,
        xLocal,
        tilingData->lastFullRepeats,
        tilingData->lastTail);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    if ((tilingData->flags & FLAG_LAST_COPY_ALIGNED) != 0U) {
        AscendC::DataCopy(yGm, yLocal, count);
    } else {
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = count * sizeof(T);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopyPad(yGm, yLocal, p);
    }
}

} // namespace NsSquare

#endif // SQUARE_H
