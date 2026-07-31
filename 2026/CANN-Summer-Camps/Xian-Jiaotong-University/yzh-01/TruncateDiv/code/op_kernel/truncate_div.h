/*!
 * \file truncate_div.h
 * \brief TruncateDiv 算子 kernel 类定义
 *
 * TruncateDiv 算子功能：
 *   y = trunc(x1 / x2)  (逐元素除法后向零取整)
 *
 * 支持广播：当 x1/x2 形状不同时，按广播规则获取对应元素。
 */

#ifndef TRUNCATEDIV_H
#define TRUNCATEDIV_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_div_tiling_data.h"
#include "truncate_div_tiling_key.h"

namespace NsTruncateDiv {

using namespace AscendC;

// ---------------------------------------------------------------------------
// bf16 类型检测 trait（无需额外头文件）
// ---------------------------------------------------------------------------
namespace detail {
    template<typename A, typename B>
    struct IsSame { static constexpr bool value = false; };
    template<typename A>
    struct IsSame<A, A> { static constexpr bool value = true; };
}  // namespace detail

template <typename T>
class TruncateDiv {
public:
    // bf16/fp16 支持：使用 fp32 中间计算，避免 Div 在低精度下整数边界舍入误差
    static constexpr bool kUseCast = detail::IsSame<T, bfloat16_t>::value || detail::IsSame<T, half>::value;
    __aicore__ inline TruncateDiv(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateDivTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 同 shape 快速路径（向量化，TQue 双缓冲）
    __aicore__ inline void CopyInFast(int64_t offset, int64_t currentNum);
    __aicore__ inline void ComputeFast(int64_t currentNum);
    __aicore__ inline void CopyOutFast(int64_t offset, int64_t currentNum);

    // 广播路径（分块 gather + 批量计算）
    __aicore__ inline void ProcessBroadcast(int64_t startOffset, int64_t thisCoreNum);

    // 将 flat output index 映射为广播后的 flat input index
    __aicore__ inline int64_t MapIndex(int64_t outIdx, const int64_t* shape, int64_t dimNum);

private:
    TPipe pipe;

    // 快速路径：TQue 双缓冲队列
    TQue<QuePosition::VECIN, 2> inQueueX1;
    TQue<QuePosition::VECIN, 2> inQueueX2;
    TQue<QuePosition::VECOUT, 2> outQueueY;

    // Trunc 高阶 API 临时缓冲区（快速路径和广播路径共用）
    TBuf<TPosition::VECCALC> truncTmpBuf;

    // bf16 路径专用：Cast 中间 fp32 缓冲区（仅 kUseCast == true 时初始化）
    TBuf<TPosition::VECCALC> castX1Fp32Buf;
    TBuf<TPosition::VECCALC> castX2Fp32Buf;
    TBuf<TPosition::VECCALC> castYFp32Buf;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t blockIdx_ = 0;
    int64_t thisCoreNum_ = 0;
    int64_t tmpSize_ = 0;

    // 广播参数
    bool needBroadcast_ = false;
    int64_t x1ElemNum_ = 0;
    int64_t x2ElemNum_ = 0;
    int64_t dimNum_ = 0;
    int64_t outShape_[kMaxDim];
    int64_t x1Shape_[kMaxDim];
    int64_t x2Shape_[kMaxDim];
};

// ============================================================================

template <typename T>
__aicore__ inline void TruncateDiv<T>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateDivTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    ubFactor_ = tilingData->ubFactor;
    x1ElemNum_ = tilingData->x1ElemNum;
    x2ElemNum_ = tilingData->x2ElemNum;
    dimNum_ = tilingData->dimNum;
    tmpSize_ = tilingData->tmpSize;

    for (int64_t i = 0; i < dimNum_; i++) {
        outShape_[i] = tilingData->outShape[i];
        x1Shape_[i] = tilingData->x1Shape[i];
        x2Shape_[i] = tilingData->x2Shape[i];
    }

    needBroadcast_ = (x1ElemNum_ != totalNum_ || x2ElemNum_ != totalNum_);

    blockIdx_ = GetBlockIdx();

    int64_t startOffset = blockIdx_ * blockFactor_;
    int64_t endOffset = startOffset + blockFactor_;
    if (endOffset > totalNum_) {
        endOffset = totalNum_;
    }
    thisCoreNum_ = (endOffset > startOffset) ? (endOffset - startOffset) : 0;

    if (thisCoreNum_ <= 0) {
        return;
    }

    // GM 地址
    inputGMX1.SetGlobalBuffer((__gm__ T*)x1);
    inputGMX2.SetGlobalBuffer((__gm__ T*)x2);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    if (!needBroadcast_) {
        // 快速路径：调整 ubFactor，初始化 TQue 队列
        if (ubFactor_ > thisCoreNum_) {
            ubFactor_ = thisCoreNum_;
        }
        int64_t bufSize = ubFactor_ * sizeof(T);
        pipe.InitBuffer(inQueueX1, 2, bufSize);
        pipe.InitBuffer(inQueueX2, 2, bufSize);
        pipe.InitBuffer(outQueueY, 2, bufSize);

        // bf16 路径：额外 fp32 中间缓冲区
        if constexpr (kUseCast) {
            pipe.InitBuffer(castX1Fp32Buf, ubFactor_ * sizeof(float));
            pipe.InitBuffer(castX2Fp32Buf, ubFactor_ * sizeof(float));
            pipe.InitBuffer(castYFp32Buf, ubFactor_ * sizeof(float));
        }
    }

    // Trunc 临时缓冲区（快速路径和广播路径共用）
    if (tmpSize_ > 0) {
        pipe.InitBuffer(truncTmpBuf, static_cast<uint32_t>(tmpSize_));
    }
}

template <typename T>
__aicore__ inline void TruncateDiv<T>::Process()
{
    if (thisCoreNum_ <= 0) {
        return;
    }

    int64_t startOffset = blockIdx_ * blockFactor_;

    if (needBroadcast_) {
        ProcessBroadcast(startOffset, thisCoreNum_);
        return;
    }

    // 同 shape 快速路径：双缓冲流水线
    int64_t processed = 0;
    bool firstTile = true;
    int64_t prevOffset = 0;
    int64_t prevNum = 0;

    while (processed < thisCoreNum_) {
        int64_t currentNum = (processed + ubFactor_ <= thisCoreNum_) ? ubFactor_ : (thisCoreNum_ - processed);
        int64_t offset = startOffset + processed;

        // 启动当前 tile 的 CopyIn（与上一个 tile 的 Compute/CopyOut 重叠）
        CopyInFast(offset, currentNum);

        if (!firstTile) {
            ComputeFast(prevNum);
            CopyOutFast(prevOffset, prevNum);
        }

        prevOffset = offset;
        prevNum = currentNum;
        firstTile = false;
        processed += currentNum;
    }

    // 最后一个 tile 的 Compute 和 CopyOut
    if (!firstTile) {
        ComputeFast(prevNum);
        CopyOutFast(prevOffset, prevNum);
    }
}

// ============================================================================
// 同 shape 快速路径
// ============================================================================

template <typename T>
__aicore__ inline void TruncateDiv<T>::CopyInFast(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
    LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    DataCopyPad(x1Local, inputGMX1[offset], copyParams, padParams);
    DataCopyPad(x2Local, inputGMX2[offset], copyParams, padParams);

    inQueueX1.EnQue(x1Local);
    inQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void TruncateDiv<T>::ComputeFast(int64_t currentNum)
{
    if constexpr (kUseCast) {
        // bf16 路径：DeQue bf16 → Cast to fp32 → 计算 → Cast back → EnQue bf16
        LocalTensor<T> x1Bf16 = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Bf16 = inQueueX2.DeQue<T>();

        // bf16 → fp32（低精度转高精度，无需舍入）
        LocalTensor<float> x1Fp32 = castX1Fp32Buf.Get<float>(currentNum);
        LocalTensor<float> x2Fp32 = castX2Fp32Buf.Get<float>(currentNum);
        Cast<float, T>(x1Fp32, x1Bf16, RoundMode::CAST_NONE, currentNum);
        Cast<float, T>(x2Fp32, x2Bf16, RoundMode::CAST_NONE, currentNum);

        inQueueX1.FreeTensor(x1Bf16);
        inQueueX2.FreeTensor(x2Bf16);

        // fp32 计算：y = trunc(x1 / x2)
        LocalTensor<float> yFp32 = castYFp32Buf.Get<float>(currentNum);
        Div(yFp32, x1Fp32, x2Fp32, currentNum);

        LocalTensor<uint8_t> tmpBuf = truncTmpBuf.Get<uint8_t>();
        Trunc<float>(yFp32, yFp32, tmpBuf, static_cast<uint32_t>(currentNum));

        // fp32 → bf16（高精度转低精度，需舍入）
        LocalTensor<T> yBf16 = outQueueY.AllocTensor<T>();
        Cast<T, float>(yBf16, yFp32, RoundMode::CAST_ROUND, currentNum);

        outQueueY.EnQue<T>(yBf16);
    } else {
        // fp16 / fp32 原始路径：同类型直接计算
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // y = trunc(x1 / x2)
        Div(yLocal, x1Local, x2Local, currentNum);

        LocalTensor<uint8_t> tmpBuf = truncTmpBuf.Get<uint8_t>();
        Trunc<T>(yLocal, yLocal, tmpBuf, static_cast<uint32_t>(currentNum));

        outQueueY.EnQue<T>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
}

template <typename T>
__aicore__ inline void TruncateDiv<T>::CopyOutFast(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[offset], yLocal, copyParams);

    outQueueY.FreeTensor(yLocal);
}

// ============================================================================
// 广播路径：分块 gather + 批量计算
// 使用 TBuf + SetFlag/WaitFlag 同步，避免 TQue 单元素操作开销
// ============================================================================

template <typename T>
__aicore__ inline int64_t TruncateDiv<T>::MapIndex(int64_t outIdx, const int64_t* shape, int64_t dimNum)
{
    if (dimNum <= 0) return 0;

    int64_t idx = 0;
    int64_t stride = 1;
    for (int64_t d = dimNum - 1; d >= 0; d--) {
        int64_t coord = outIdx % outShape_[d];
        outIdx /= outShape_[d];
        if (shape[d] > 1) {
            idx += coord * stride;
        }
        stride *= shape[d];
    }
    return idx;
}

template <typename T>
__aicore__ inline void TruncateDiv<T>::ProcessBroadcast(int64_t startOffset, int64_t thisCoreNum)
{
    constexpr int64_t BROADCAST_BLOCK = 256;

    // 广播路径使用 TBuf + SetFlag/WaitFlag 同步
    TBuf<TPosition::VECCALC> x1BroadBuf;
    TBuf<TPosition::VECCALC> x2BroadBuf;
    TBuf<TPosition::VECCALC> yBroadBuf;

    pipe.InitBuffer(x1BroadBuf, BROADCAST_BLOCK * sizeof(T));
    pipe.InitBuffer(x2BroadBuf, BROADCAST_BLOCK * sizeof(T));
    pipe.InitBuffer(yBroadBuf, BROADCAST_BLOCK * sizeof(T));

    // bf16 路径：额外 fp32 中间缓冲区
    TBuf<TPosition::VECCALC> x1BroadFp32, x2BroadFp32, yBroadFp32;
    if constexpr (kUseCast) {
        pipe.InitBuffer(x1BroadFp32, BROADCAST_BLOCK * sizeof(float));
        pipe.InitBuffer(x2BroadFp32, BROADCAST_BLOCK * sizeof(float));
        pipe.InitBuffer(yBroadFp32, BROADCAST_BLOCK * sizeof(float));
    }

    LocalTensor<T> x1Local = x1BroadBuf.Get<T>(BROADCAST_BLOCK);
    LocalTensor<T> x2Local = x2BroadBuf.Get<T>(BROADCAST_BLOCK);
    LocalTensor<T> yLocal = yBroadBuf.Get<T>(BROADCAST_BLOCK);

    // 单元素搬运参数
    DataCopyExtParams copyParams1{1, sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t blockStart = 0; blockStart < thisCoreNum; blockStart += BROADCAST_BLOCK) {
        int64_t curBlock = (blockStart + BROADCAST_BLOCK <= thisCoreNum) ? BROADCAST_BLOCK : (thisCoreNum - blockStart);

        // Gather: 按广播规则从 GM 取 x1 元素到 local buffer
        for (int64_t j = 0; j < curBlock; j++) {
            int64_t outIdx = startOffset + blockStart + j;
            int64_t x1Idx = MapIndex(outIdx, x1Shape_, dimNum_);
            DataCopyPad(x1Local[j], inputGMX1[x1Idx], copyParams1, padParams);
        }

        // Gather: 按广播规则从 GM 取 x2 元素到 local buffer
        for (int64_t j = 0; j < curBlock; j++) {
            int64_t outIdx = startOffset + blockStart + j;
            int64_t x2Idx = MapIndex(outIdx, x2Shape_, dimNum_);
            DataCopyPad(x2Local[j], inputGMX2[x2Idx], copyParams1, padParams);
        }

        // 等待 DMA 完成
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        // y = trunc(x1 / x2)
        if constexpr (kUseCast) {
            // bf16 路径：Cast bf16 → fp32 → 计算 → Cast fp32 → bf16
            LocalTensor<float> x1Fp32 = x1BroadFp32.Get<float>(BROADCAST_BLOCK);
            LocalTensor<float> x2Fp32 = x2BroadFp32.Get<float>(BROADCAST_BLOCK);
            LocalTensor<float> yFp32 = yBroadFp32.Get<float>(BROADCAST_BLOCK);

            Cast<float, T>(x1Fp32, x1Local, RoundMode::CAST_NONE, curBlock);
            Cast<float, T>(x2Fp32, x2Local, RoundMode::CAST_NONE, curBlock);

            Div(yFp32, x1Fp32, x2Fp32, curBlock);

            LocalTensor<uint8_t> tmpBuf = truncTmpBuf.Get<uint8_t>();
            Trunc<float>(yFp32, yFp32, tmpBuf, static_cast<uint32_t>(curBlock));

            Cast<T, float>(yLocal, yFp32, RoundMode::CAST_ROUND, curBlock);
        } else {
            Div(yLocal, x1Local, x2Local, curBlock);

            LocalTensor<uint8_t> tmpBuf = truncTmpBuf.Get<uint8_t>();
            Trunc<T>(yLocal, yLocal, tmpBuf, static_cast<uint32_t>(curBlock));
        }

        // 等待计算完成
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        // Scatter: 将结果逐个写回 GM
        DataCopyExtParams copyParamsOut{1, sizeof(T), 0, 0, 0};
        for (int64_t j = 0; j < curBlock; j++) {
            int64_t outIdx = startOffset + blockStart + j;
            DataCopyPad(outputGMY[outIdx], yLocal[j], copyParamsOut);
        }
    }

    PipeBarrier<PIPE_ALL>();
}

} // namespace NsTruncateDiv
#endif // TRUNCATEDIV_H
