/*!
 * \file square.h
 * \brief Square 算子 Kernel 类定义
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

// 手动同步事件
constexpr event_t EVENT_MTE2_0 = EVENT_ID0;
constexpr event_t EVENT_MTE2_1 = EVENT_ID1;
constexpr event_t EVENT_V_0    = EVENT_ID2;
constexpr event_t EVENT_V_1    = EVENT_ID3;

constexpr int64_t DATA_BLOCK_BYTES = 32;

template <typename T>
class Square {
public:
    __aicore__ inline Square() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const SquareTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<T>& dst,
        int64_t gmOffset,
        int64_t elemNum);

    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<T>& src,
        int64_t gmOffset,
        int64_t elemNum);

    __aicore__ inline void Compute(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T>& src,
        int64_t ubLen);

private:
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator_;

    int64_t blockLength_ = 0;   // 当前核有效元素数
    int64_t alignNum_ = 0;      // 32 字节对应的元素数
};

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SquareTilingData* tilingData)
{
    alignNum_ = DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));

    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    const int64_t remaining = tilingData->totalNum - blockOffset;
    blockLength_ = (remaining <= 0) ? 0
                  : (remaining < blockFactor ? remaining : blockFactor);

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    AscendC::LocalTensor<T>& dst,
    int64_t gmOffset,
    int64_t elemNum)
{
    if (elemNum % alignNum_ == 0) {
        AscendC::DataCopy(dst, inputGm_[gmOffset], elemNum);
    } else {
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(elemNum * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, inputGm_[gmOffset], copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void Square<T>::Compute(
    AscendC::LocalTensor<T>& dst,
    AscendC::LocalTensor<T>& src,
    int64_t ubLen)
{
    AscendC::Mul(dst, src, src, static_cast<int32_t>(ubLen));
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    AscendC::LocalTensor<T>& src,
    int64_t gmOffset,
    int64_t elemNum)
{
    if (elemNum % alignNum_ == 0) {
        AscendC::DataCopy(outputGm_[gmOffset], src, elemNum);
    } else {
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(elemNum * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(outputGm_[gmOffset], src, copyParams);
    }
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ == 0 || alignNum_ == 0) {
        return;
    }

    // 每核切成 2 个 tile，长度按 32 字节对齐向下取
    int64_t tileLen = (blockLength_ / 2 / alignNum_) * alignNum_;
    if (tileLen == 0) {
        // 数据量太小，退化成单 tile
        int64_t ubLen = ((blockLength_ + alignNum_ - 1) / alignNum_) * alignNum_;
        AscendC::LocalTensor<T> xLocal = ubAllocator_.Alloc<T>(ubLen);
        AscendC::LocalTensor<T> yLocal = ubAllocator_.Alloc<T>(ubLen);

        CopyIn(xLocal, 0, blockLength_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_0);

        Compute(yLocal, xLocal, ubLen);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_0);

        CopyOut(yLocal, 0, blockLength_);
        return;
    }

    int64_t tile0Len = tileLen;
    int64_t tile1Len = blockLength_ - tile0Len;
    int64_t tile0Ub = tile0Len; // 已对齐
    int64_t tile1Ub = ((tile1Len + alignNum_ - 1) / alignNum_) * alignNum_;

    // ping-pong 四份 UB
    AscendC::LocalTensor<T> x0 = ubAllocator_.Alloc<T>(tile0Ub);
    AscendC::LocalTensor<T> y0 = ubAllocator_.Alloc<T>(tile0Ub);
    AscendC::LocalTensor<T> x1 = ubAllocator_.Alloc<T>(tile1Ub);
    AscendC::LocalTensor<T> y1 = ubAllocator_.Alloc<T>(tile1Ub);

    // ---------- tile 0 ----------
    CopyIn(x0, 0, tile0Len);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_0);

    // ---------- tile 0 计算 ----------
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_0);
    Compute(y0, x0, tile0Ub);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_0);

    // ---------- tile 1 搬入（与 tile0 计算/回写重叠） ----------
    CopyIn(x1, tile0Len, tile1Len);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_1);

    // ---------- tile 1 计算（与 tile0 回写重叠） ----------
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_1);
    Compute(y1, x1, tile1Ub);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_1);

    // ---------- tile 0 回写 ----------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_0);
    CopyOut(y0, 0, tile0Len);

    // ---------- tile 1 回写 ----------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_1);
    CopyOut(y1, tile0Len, tile1Len);
}

} // namespace NsSquare

#endif // SQUARE_H
