/**
 * @file inplace_rsqrt.h
 * @brief InplaceRsqrt（原地 1/sqrt(x)）AscendC Kernel 实现
 *
 * 功能 : out = 1 / sqrt(self)，结果原位覆盖输入 self（输入、输出是同一块 GM 显存）
 * shape: 4 维 [N4, N3, N2, N1]，Format = ND，按连续内存展平后处理
 * dtype: float16 (half)
 * 计算 : AscendC::Rsqrt（底层映射到 vrsqrt 向量指令），批量向量化，无标量循环
 *
 * 流水结构（TQueBind 原地模式）：
 *   DataCopy(GM->UB) -> EnQue -> DeQue -> Rsqrt(UB 内原地) -> EnQue -> DeQue -> DataCopy(UB->GM 同地址)
 *
 * 为什么用 TQueBind 而不是 TQue：
 *   原地算子的输入 Buffer 同时充当输出 Buffer，TQueBind<VECIN, VECOUT, N> 允许同一块 UB
 *   既作为 VECIN 入队又作为 VECOUT 出队，从而在不额外占用 UB 的前提下保留双缓冲流水能力。
 *
 * 对齐与尾块：
 *   GM<->UB 的 DataCopy 要求 32Byte 对齐；非对齐长度的尾块统一用 DataCopyPad 处理，
 *   因此 4 维 shape 的最后一维 N1 不需要是 16 的整数倍。
 */
#ifndef INPLACE_RSQRT_KERNEL_H
#define INPLACE_RSQRT_KERNEL_H

#include "kernel_operator.h"

namespace InplaceRsqrtNs {
using namespace AscendC;

// UB 队列深度：2 -> 搬运 / 计算 / 搬出 三级流水（Double Buffer）
constexpr int32_t BUFFER_NUM = 2;

// DataCopy 的对齐粒度：GM <-> UB 搬运要求 32 Byte 对齐
constexpr uint32_t BYTE_ALIGN = 32;

// 是否在 vrsqrt 之后追加一次牛顿迭代收敛：
//   y1 = y0 * (1.5 - 0.5 * x * y0 * y0)
// 默认关闭；硬件 rsqrt 为近似实现，打开后精度显著提升（代价：2 块 UB 临时缓冲 + 5 条向量指令）
#ifndef INPLACE_RSQRT_NEWTON_REFINE
#define INPLACE_RSQRT_NEWTON_REFINE 0
#endif

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (b == 0) ? a : (a + b - 1) / b;
}

__aicore__ inline uint32_t CeilAlignU32(uint32_t a, uint32_t b)
{
    return (b == 0) ? a : (a + b - 1) / b * b;
}

template <typename T>
class InplaceRsqrt {
    static_assert(IsSameType<T, half>::Value || IsSameType<T, float>::Value,
                  "InplaceRsqrt only supports float16(half) / float32(float)");

public:
    __aicore__ inline InplaceRsqrt() {}

    /**
     * @brief 初始化
     * @param self        输入/输出同一块 GM 地址（原地）
     * @param selfOut     输出 GM 地址：与 self 相同时即为原地（常见情况）；
     *                    若传入 nullptr 或与 self 相同，跳过额外的输出搬运
     * @param totalLength 元素总个数（4 维 shape 展平后的元素数）
     * @param tileLength  单次 DataCopy 的元素个数（建议 32B 对齐，如 8192）
     */
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR selfOut, uint32_t totalLength, uint32_t tileLength)
    {
        constexpr uint32_t alignElem = BYTE_ALIGN / sizeof(T);  // half: 16 个元素 = 32B

        blockNum_ = static_cast<uint32_t>(GetBlockNum());
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        if (blockNum_ == 0) {
            blockNum_ = 1;
        }

        // tile 长度必须 32B 对齐，否则 DataCopy 会越界踩内存
        tileLength_ = CeilAlignU32(tileLength, alignElem);
        if (tileLength_ == 0) {
            tileLength_ = alignElem;
        }

        // UB 申请：原地算子只需要一条队列（同一块 UB 既是 VECIN 也是 VECOUT）
        pipe_.InitBuffer(selfQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
#if INPLACE_RSQRT_NEWTON_REFINE
        pipe_.InitBuffer(yBuf_, tileLength_ * sizeof(T));
        pipe_.InitBuffer(uBuf_, tileLength_ * sizeof(T));
#endif

        coreOffset_ = 0;
        coreSize_ = 0;
        if (totalLength == 0 || blockIdx_ >= blockNum_) {
            return;
        }

        // 多核切分：每个核处理一段连续数据；段首地址同样要 32B 对齐，故 perCore 向上对齐
        uint32_t perCore = CeilAlignU32(CeilDivU32(totalLength, blockNum_), alignElem);
        uint64_t offset = static_cast<uint64_t>(blockIdx_) * perCore;
        if (offset >= totalLength) {
            return;  // 本核无数据，直接退出
        }
        coreOffset_ = static_cast<uint32_t>(offset);
        uint32_t remain = totalLength - coreOffset_;
        coreSize_ = (remain < perCore) ? remain : perCore;

        // 原地：直接把输入 GM 当作输出 GM，不做任何额外的搬运
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self) + coreOffset_, coreSize_);
        aliasOut_ = (selfOut == nullptr) || (selfOut == self);
        if (!aliasOut_) {
            outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(selfOut) + coreOffset_, coreSize_);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreSize_ == 0) {
            return;
        }
        uint32_t offset = 0;
        uint32_t remain = coreSize_;
        while (remain > 0) {
            // 中间 tile 恒为 32B 对齐长度；只有全张量的最后一个 tile 可能非对齐
            uint32_t cur = (remain < tileLength_) ? remain : tileLength_;
            CopyIn(offset, cur);
            Compute(cur);
            CopyOut(offset, cur);
            offset += cur;
            remain -= cur;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        constexpr uint32_t alignElem = BYTE_ALIGN / sizeof(T);
        LocalTensor<T> x = selfQueue_.AllocTensor<T>();
        if ((count % alignElem) == 0) {
            DataCopy(x, selfGm_[offset], count);
        } else {
            // 尾块：长度非 32B 对齐，用 DataCopyPad 避免越界读写
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(x, selfGm_[offset], copyParams, padParams);
        }
        selfQueue_.EnQue(x);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> x = selfQueue_.DeQue<T>();
#if INPLACE_RSQRT_NEWTON_REFINE
        LocalTensor<T> y = yBuf_.Get<T>();   // y0 = 1/sqrt(x)
        LocalTensor<T> u = uBuf_.Get<T>();   // 修正因子
        Rsqrt(y, x, count);                              // y0 = vrsqrt(x)
        Mul(u, y, y, count);                             // u = y0 * y0
        Mul(u, u, x, count);                             // u = x * y0^2
        Muls(u, u, static_cast<T>(-0.5f), count);        // u = -0.5 * x * y0^2
        Adds(u, u, static_cast<T>(1.5f), count);         // u = 1.5 - 0.5 * x * y0^2
        Mul(x, y, u, count);                             // 结果就地写回队列缓冲
#else
        // dst 与 src 为同一块 UB：逐元素向量指令无数据相关，支持原地
        Rsqrt(x, x, count);
#endif
        selfQueue_.EnQue(x);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> x = selfQueue_.DeQue<T>();
        CopyOutOne(selfGm_, offset, count, x);   // 写回原地址 => 原地生效
        if (!aliasOut_) {
            CopyOutOne(outGm_, offset, count, x);
        }
        selfQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOutOne(const GlobalTensor<T> &dst, uint32_t offset, uint32_t count,
                                      const LocalTensor<T> &x)
    {
        constexpr uint32_t alignElem = BYTE_ALIGN / sizeof(T);
        if ((count % alignElem) == 0) {
            DataCopy(dst[offset], x, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst[offset], x, copyParams);
        }
    }

private:
    TPipe pipe_;
    // 输入 Buffer 同时作为输出 Buffer
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> selfQueue_;
#if INPLACE_RSQRT_NEWTON_REFINE
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> uBuf_;
#endif
    GlobalTensor<T> selfGm_;   // 输入（=原地输出）
    GlobalTensor<T> outGm_;    // 仅当框架给输出分配了独立地址时使用

    uint32_t blockNum_ = 1;
    uint32_t blockIdx_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreSize_ = 0;
    bool aliasOut_ = true;
};

}  // namespace InplaceRsqrtNs

#endif  // INPLACE_RSQRT_KERNEL_H