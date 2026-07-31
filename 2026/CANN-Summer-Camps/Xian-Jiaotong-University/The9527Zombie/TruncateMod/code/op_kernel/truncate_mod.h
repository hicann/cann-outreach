/*!
 * \file truncate_mod.h
 * \brief TruncateMod kernel class (Ascend C, A2/A3).
 *
 * Element-wise truncated remainder: y = x1 - trunc(x1 / x2) * x2, where trunc()
 * rounds toward zero, so the result keeps the sign of the dividend x1 (identical
 * to numpy / torch fmod).
 *
 * Every dtype is promoted to float on the UB for the core computation:
 *   trunc(t) = floor(max(t, 0)) + ceil(min(t, 0))    // no int overflow
 * Division-by-zero follows the built-in operator's golden:
 *   float  : x2 == 0  ->  NaN
 *   int    : x2 == 0  ->  -1   (0xFF... bit pattern)
 *   uint   : x2 == 0  ->  255  (0xFF)
 *
 * Phase 1 scope: same-shape (element-wise) inputs. Broadcast is added in a later
 * phase; the tiling already carries the broadcast strides for that work.
 * Note: int32 uses the float path and is exact for |value| <= 2^24; exact
 * large-magnitude int32 is hardened together with int64 in a later phase.
 */
#ifndef TRUNCATEMOD_H
#define TRUNCATEMOD_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"

namespace NsTruncateMod {

using namespace AscendC;

template <typename T>
class TruncateModKernel {
public:
    __aicore__ inline explicit TruncateModKernel(const TruncateModTilingData* tilingData)
        : coreNum(tilingData->coreNum),
          bufferNum(tilingData->bufferNum),
          tileCount(tilingData->tileCount),
          perCoreCount(tilingData->perCoreCount),
          tailCoreNum(tilingData->tailCoreNum),
          lastCoreCount(tilingData->lastCoreCount),
          blockElem(tilingData->blockElem)
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < this->tailCoreNum) {
            this->myCount = this->perCoreCount + this->blockElem;
            this->myOffset = blockIdx * this->myCount;
        } else {
            this->myOffset = this->tailCoreNum * (this->perCoreCount + this->blockElem) +
                             (blockIdx - this->tailCoreNum) * this->perCoreCount;
            this->myCount = this->perCoreCount;
        }
        if (blockIdx == this->coreNum - 1u) {
            this->myCount = this->lastCoreCount;
        }

        this->pipe.InitBuffer(this->inQue1, this->bufferNum, this->tileCount * sizeof(T));
        this->pipe.InitBuffer(this->inQue2, this->bufferNum, this->tileCount * sizeof(T));
        this->pipe.InitBuffer(this->outQue, this->bufferNum, this->tileCount * sizeof(T));
        this->pipe.InitBuffer(this->x1fBuf, this->tileCount * sizeof(float));
        this->pipe.InitBuffer(this->x2fBuf, this->tileCount * sizeof(float));
        this->pipe.InitBuffer(this->quotBuf, this->tileCount * sizeof(float));
        this->pipe.InitBuffer(this->tmpBuf, this->tileCount * sizeof(float));
        this->pipe.InitBuffer(this->maskBuf, this->tileCount * sizeof(uint8_t));
        if constexpr (kNeedHalf) {
            this->pipe.InitBuffer(this->halfBuf, this->tileCount * sizeof(half));
        }
    }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        this->x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        this->x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        this->yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    }

    __aicore__ inline void Process()
    {
        uint64_t remaining = this->myCount;
        uint64_t offset = this->myOffset;
        while (remaining > 0u) {
            uint64_t cnt = (remaining < this->tileCount) ? remaining : this->tileCount;
            this->CopyIn(offset, cnt);
            this->Compute(cnt);
            this->CopyOut(offset, cnt);
            offset += cnt;
            remaining -= cnt;
        }
    }

private:
    static constexpr bool kIsFloatT =
        std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float>;
    static constexpr bool kIsSmallInt = std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>;
    static constexpr bool kNeedHalf = kIsSmallInt; // int8/uint8 route through half for the float cast
    static constexpr bool kIsUnsigned = std::is_same_v<T, uint8_t>;

    __aicore__ inline void CopyIn(uint64_t offset, uint64_t cnt)
    {
        AscendC::DataCopyExtParams params{1u, static_cast<uint32_t>(cnt * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<T> pad{false, 0u, 0u, static_cast<T>(0)};

        AscendC::LocalTensor<T> x1L = this->inQue1.template AllocTensor<T>();
        AscendC::DataCopyPad(x1L, this->x1Global[offset], params, pad);
        this->inQue1.template EnQue<T>(x1L);

        AscendC::LocalTensor<T> x2L = this->inQue2.template AllocTensor<T>();
        AscendC::DataCopyPad(x2L, this->x2Global[offset], params, pad);
        this->inQue2.template EnQue<T>(x2L);
    }

    // Cast raw input (any dtype) to fp32 in dst.
    __aicore__ inline void ToFloat(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<T>& src,
                                   uint64_t cnt)
    {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(dst, src.template ReinterpretCast<float>(), this->AlignUp(cnt));
        } else if constexpr (kIsSmallInt) {
            AscendC::LocalTensor<half> h = this->halfBuf.template Get<half>();
            AscendC::Cast<half, T>(h, src, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Cast<float, half>(dst, h, AscendC::RoundMode::CAST_NONE, cnt);
        } else {
            // half / bfloat16 / int32
            AscendC::Cast<float, T>(dst, src, AscendC::RoundMode::CAST_NONE, cnt);
        }
    }

    // Cast fp32 result back to T in dst.
    __aicore__ inline void FromFloat(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<float>& src,
                                     uint64_t cnt)
    {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(dst.template ReinterpretCast<float>(), src, this->AlignUp(cnt));
        } else if constexpr (std::is_same_v<T, half>) {
            AscendC::Cast<half, float>(dst, src, AscendC::RoundMode::CAST_NONE, cnt);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::Cast<bfloat16_t, float>(dst, src, AscendC::RoundMode::CAST_RINT, cnt);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            AscendC::Cast<int32_t, float>(dst, src, AscendC::RoundMode::CAST_RINT, cnt);
        } else {
            // int8 / uint8: float -> half -> int
            AscendC::LocalTensor<half> h = this->halfBuf.template Get<half>();
            AscendC::Cast<half, float>(h, src, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Cast<T, half>(dst, h, AscendC::RoundMode::CAST_RINT, cnt);
        }
    }

    __aicore__ inline void Compute(uint64_t cnt)
    {
        AscendC::LocalTensor<T> x1L = this->inQue1.template DeQue<T>();
        AscendC::LocalTensor<T> x2L = this->inQue2.template DeQue<T>();
        AscendC::LocalTensor<float> x1f = this->x1fBuf.template Get<float>();
        AscendC::LocalTensor<float> x2f = this->x2fBuf.template Get<float>();
        AscendC::LocalTensor<float> quot = this->quotBuf.template Get<float>();
        AscendC::LocalTensor<float> tmp = this->tmpBuf.template Get<float>();

        this->ToFloat(x1f, x1L, cnt);
        this->ToFloat(x2f, x2L, cnt);
        this->inQue1.template FreeTensor<T>(x1L);
        this->inQue2.template FreeTensor<T>(x2L);

        // q = x1 / x2 ; trunc(q) = floor(max(q,0)) + ceil(min(q,0))
        AscendC::Div<float>(quot, x1f, x2f, cnt);
        AscendC::Mins<float>(tmp, quot, 0.0f, cnt);
        AscendC::Ceil<float>(tmp, tmp, cnt);
        AscendC::Maxs<float>(quot, quot, 0.0f, cnt);
        AscendC::Floor<float>(quot, quot, cnt);
        AscendC::Add<float>(quot, quot, tmp, cnt); // trunc(q)
        AscendC::Mul<float>(quot, quot, x2f, cnt); // trunc(q) * x2
        AscendC::Sub<float>(x1f, x1f, quot, cnt);  // remainder in x1f

        this->ApplyDivZero(x1f, x2f, tmp, cnt);

        AscendC::LocalTensor<T> dstL = this->outQue.template AllocTensor<T>();
        this->FromFloat(dstL, x1f, cnt);
        this->outQue.template EnQue<T>(dstL);
    }

    // Overwrite positions where x2 == 0 with the built-in operator's sentinel.
    __aicore__ inline void ApplyDivZero(const AscendC::LocalTensor<float>& rem, const AscendC::LocalTensor<float>& x2f,
                                        const AscendC::LocalTensor<float>& tmp, uint64_t cnt)
    {
        AscendC::LocalTensor<uint8_t> mask = this->maskBuf.template Get<uint8_t>();
        float sentinel;
        if constexpr (kIsFloatT) {
            union {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = 0x7FC00000u; // quiet NaN bit pattern
            sentinel = cvt.f;
        } else if constexpr (kIsUnsigned) {
            sentinel = 255.0f;
        } else {
            sentinel = -1.0f;
        }
        AscendC::Duplicate<float>(tmp, sentinel, cnt);
        AscendC::CompareScalar(mask, x2f, 0.0f, AscendC::CMPMODE::EQ, this->AlignUp(cnt));
        AscendC::Select<float>(rem, mask, tmp, rem, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->AlignUp(cnt));
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t cnt)
    {
        AscendC::LocalTensor<T> dstL = this->outQue.template DeQue<T>();
        AscendC::DataCopyExtParams params{1u, static_cast<uint32_t>(cnt * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPad(this->yGlobal[offset], dstL, params);
        this->outQue.template FreeTensor<T>(dstL);
    }

    __aicore__ inline uint64_t AlignUp(uint64_t n)
    {
        constexpr uint64_t kAlign = 64u; // fp32 vector repeat
        return (n + kAlign - 1u) & ~(kAlign - 1u);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQue1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQue2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x1fBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2fBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> quotBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> halfBuf;

    AscendC::GlobalTensor<T> x1Global;
    AscendC::GlobalTensor<T> x2Global;
    AscendC::GlobalTensor<T> yGlobal;

    uint64_t coreNum = 1u;
    uint64_t bufferNum = 1u;
    uint64_t tileCount = 0u;
    uint64_t perCoreCount = 0u;
    uint64_t tailCoreNum = 0u;
    uint64_t lastCoreCount = 0u;
    uint64_t blockElem = 0u;
    uint64_t myOffset = 0u;
    uint64_t myCount = 0u;
};

template <typename T>
__aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData)
{
    TruncateModKernel<T> op(tilingData);
    op.Init(x1, x2, y);
    op.Process();
}

} // namespace NsTruncateMod

#endif // TRUNCATEMOD_H
