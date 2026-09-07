/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义 - v4 static-tensor optimized
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int32_t EVENT_ID_PING = 0;
constexpr int32_t EVENT_ID_PONG = 1;

template <typename T>
class Square {
public:
    __aicore__ inline Square() {}

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        LocalTensor<T>& local,
        int64_t gmOffset,
        int32_t count,
        uint32_t localAddr);

    __aicore__ inline void ComputeInplace(
        LocalTensor<T>& local,
        int32_t count);

    __aicore__ inline void CopyOut(
        LocalTensor<T>& local,
        int64_t gmOffset,
        int32_t count,
        uint32_t localAddr);

    __aicore__ inline void ProcessSingleTile(int32_t count);
    __aicore__ inline void ProcessMultiTile();

private:
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int32_t ubLength_ = 0;
    uint32_t ubBytes_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
{
    const int64_t blockStart =
        static_cast<int64_t>(GetBlockIdx()) *
        tilingData->blockFactor;

    const int64_t remain =
        tilingData->totalNum - blockStart;

    if (remain > 0) {
        blockLength_ =
            (remain < tilingData->blockFactor) ?
            remain :
            tilingData->blockFactor;
    } else {
        blockLength_ = 0;
    }

    ubLength_ =
        static_cast<int32_t>(tilingData->ubFactor);

    ubBytes_ =
        static_cast<uint32_t>(ubLength_) *
        static_cast<uint32_t>(sizeof(T));

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x + blockStart,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output + blockStart,
        blockLength_);
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    LocalTensor<T>& local,
    int64_t gmOffset,
    int32_t count,
    uint32_t localAddr)
{
    constexpr int32_t ALIGN_ELEMENTS =
        static_cast<int32_t>(
            DATA_BLOCK_BYTES / sizeof(T));

    const int32_t alignedCount =
        (count / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    const int32_t tailCount =
        count - alignedCount;

    if (alignedCount > 0) {
        DataCopy(
            local,
            inputGMX[gmOffset],
            static_cast<uint32_t>(alignedCount));
    }

    // 非对齐时仅最后不足 32B 的 fragment 使用 DataCopyPad。
    if (tailCount > 0) {
        const uint32_t alignedBytes =
            static_cast<uint32_t>(alignedCount) *
            static_cast<uint32_t>(sizeof(T));

        LocalTensor<T> tailLocal(
            TPosition::VECCALC,
            localAddr + alignedBytes,
            static_cast<uint32_t>(ALIGN_ELEMENTS));

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(tailCount) *
                static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0};

        DataCopyPadExtParams<T> padParams{
            false,
            0,
            0,
            0};

        DataCopyPad(
            tailLocal,
            inputGMX[gmOffset + alignedCount],
            copyParams,
            padParams);
    }
}

template <typename T>
__aicore__ inline void Square<T>::ComputeInplace(
    LocalTensor<T>& local,
    int32_t count)
{
    // 完全地址重叠，直接原地平方，不申请 output UB。
    Mul(
        local,
        local,
        local,
        static_cast<int32_t>(count));
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    LocalTensor<T>& local,
    int64_t gmOffset,
    int32_t count,
    uint32_t localAddr)
{
    constexpr int32_t ALIGN_ELEMENTS =
        static_cast<int32_t>(
            DATA_BLOCK_BYTES / sizeof(T));

    const int32_t alignedCount =
        (count / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    const int32_t tailCount =
        count - alignedCount;

    if (alignedCount > 0) {
        DataCopy(
            outputGMY[gmOffset],
            local,
            static_cast<uint32_t>(alignedCount));
    }

    if (tailCount > 0) {
        const uint32_t alignedBytes =
            static_cast<uint32_t>(alignedCount) *
            static_cast<uint32_t>(sizeof(T));

        LocalTensor<T> tailLocal(
            TPosition::VECCALC,
            localAddr + alignedBytes,
            static_cast<uint32_t>(ALIGN_ELEMENTS));

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(tailCount) *
                static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0};

        DataCopyPad(
            outputGMY[gmOffset + alignedCount],
            tailLocal,
            copyParams);
    }
}

template <typename T>
__aicore__ inline void Square<T>::ProcessSingleTile(
    int32_t count)
{
    constexpr uint32_t LOCAL_ADDR = 0;

    LocalTensor<T> local(
        TPosition::VECCALC,
        LOCAL_ADDR,
        static_cast<uint32_t>(ubLength_));

    // 单 tile 最短路径：
    // CopyIn -> MTE2_V -> Mul -> V_MTE3 -> CopyOut
    CopyIn(local, 0, count, LOCAL_ADDR);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_PING);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_PING);

    ComputeInplace(local, count);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_PING);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_PING);

    CopyOut(local, 0, count, LOCAL_ADDR);
}

template <typename T>
__aicore__ inline void Square<T>::ProcessMultiTile()
{
    const uint32_t pingAddr = 0;
    const uint32_t pongAddr = ubBytes_;

    LocalTensor<T> ping(
        TPosition::VECCALC,
        pingAddr,
        static_cast<uint32_t>(ubLength_));

    LocalTensor<T> pong(
        TPosition::VECCALC,
        pongAddr,
        static_cast<uint32_t>(ubLength_));

    const int64_t loopCount =
        (blockLength_ +
         static_cast<int64_t>(ubLength_) - 1) /
        static_cast<int64_t>(ubLength_);

    // 手工 DoubleBuffer：MTE3_MTE2 负责同一 ping/pong
    // 跨轮次 buffer 复用依赖。
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID_PING);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID_PONG);

    int64_t offset = 0;

    for (int64_t i = 0; i < loopCount; ++i) {
        const bool usePing =
            ((i & 1) == 0);

        const int32_t eventId =
            usePing ?
            EVENT_ID_PING :
            EVENT_ID_PONG;

        const uint32_t localAddr =
            usePing ?
            pingAddr :
            pongAddr;

        LocalTensor<T>& local =
            usePing ?
            ping :
            pong;

        const int64_t remain =
            blockLength_ - offset;

        const int32_t currentCount =
            static_cast<int32_t>(
                remain <
                    static_cast<int64_t>(ubLength_) ?
                remain :
                static_cast<int64_t>(ubLength_));

        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        CopyIn(
            local,
            offset,
            currentCount,
            localAddr);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        ComputeInplace(
            local,
            currentCount);

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        CopyOut(
            local,
            offset,
            currentCount,
            localAddr);

        SetFlag<HardEvent::MTE3_MTE2>(eventId);

        offset += currentCount;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID_PING);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID_PONG);
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    if (blockLength_ <=
        static_cast<int64_t>(ubLength_)) {
        ProcessSingleTile(
            static_cast<int32_t>(blockLength_));
    } else {
        ProcessMultiTile();
    }
}

} // namespace NsSquare

#endif // SQUARE_H