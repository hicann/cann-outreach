/*!
 * \file truncate_mod.h
 * \brief TruncateMod 算子 kernel 类定义
 *
 * 设计依据：docs/DESIGN.md v1.1（§2.4 类别特有设计、§3.5 入口与类模板）。
 * 实现要点：
 *  - 三路模板分派：T=half（mode0）/ bfloat16_t（mode1）/ float（mode2），计算链恒为 FP32；
 *  - 广播加载策略 0/1/2/3（row-run 模型），策略 2 增加掩码连续模式单次迭代上限防护；
 *  - 标量输入（inTotal==1）在 Init 通过 GlobalTensor::GetValue 读取一次，CopyIn 用 Duplicate 展开；
 *  - 全模式统一走 VECOUT outQueueY 标准输出路径。
 */

#ifndef TRUNCATEMOD_H
#define TRUNCATEMOD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"
#include "truncate_mod_tiling_key.h"

namespace NsTruncateMod {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// Copy 掩码连续模式（Normal 高维切分）单次迭代可处理元素数上限 = 256B / sizeof(T)：
// 16 位类型 128、32 位类型 64（证据：docs/zh/api/SIMD-API/基础API/Memory矢量计算/SIMD计算说明/掩码.md
// 「连续模式 mask∈[0,128](16bit) / [0,64](32bit)」+ asc/impl/basic_api/kernel_utils_base.h SetMask）。
// 策略 2 的 mask=period 必须满足 period <= maskCap，否则掩码被硬件钳制产生错误结果。

template <typename T>
class TruncateMod {
public:
    __aicore__ inline TruncateMod(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    // 非标量输入 tile 加载（DESIGN.md §2.4.3 策略 1/2/3）
    __aicore__ inline void LoadTile(const GlobalTensor<T>& gm, const LocalTensor<T>& local, int64_t s,
        int64_t tileLen, int64_t innerLen, int64_t rowCount);

    // 标量输入展开：mode0/1 写 FP32 工作缓冲 xF；mode2 写输入队列缓冲
    __aicore__ inline void ExpandScalar(const GlobalTensor<T>& gm, float scalarValue, TQue<QuePosition::VECIN, BUFFER_NUM>& queue,
        TBuf<QuePosition::VECCALC>& fpBuf, int64_t currentNum, bool& enqueued);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    TBuf<QuePosition::VECCALC> x1FBuf; // FP32 工作缓冲（mode0/1 升精度后 x1；mode2 不使用）
    TBuf<QuePosition::VECCALC> x2FBuf; // FP32 工作缓冲（mode0/1 升精度后 x2；mode2 不使用）
    TBuf<QuePosition::VECCALC> qFBuf;  // FP32 商/截断/乘积工作缓冲（全模式）

    int64_t totalNum_ = 0;     // 广播后输出总元素数
    int64_t blockFactor_ = 1;  // 每核元素数（512 对齐）
    int64_t ubFactor_ = 0;     // 每次 UB 循环元素数
    int64_t x1InnerLen_ = 0;   // x1 连续运行长度
    int64_t x1RowCount_ = 0;   // x1 源行数
    int64_t x2InnerLen_ = 0;   // x2 连续运行长度
    int64_t x2RowCount_ = 0;   // x2 源行数
    int64_t x1Period_ = 1;     // x1 重复周期 = innerLen × rowCount
    int64_t x2Period_ = 1;     // x2 重复周期
    int64_t blockStart_ = 0;   // 当前核起始输出偏移
    bool x1Scalar_ = false;    // x1 是否标量输入（inTotal==1）
    bool x2Scalar_ = false;    // x2 是否标量输入
    float x1ScalarValue_ = 0.0f; // x1 标量值（float，Init 时读取）
    float x2ScalarValue_ = 0.0f; // x2 标量值
};

// 标量值读取：GlobalTensor::GetValue（S 管道），与 CANN 内置算子 fill_diagonal_v2 同款模式。
// 说明：DESIGN.md §1.2.2 原为「1 元素拷贝到 UB 再 GetValue」；本实现直接 GM GetValue，
// 省去 scalarBuf 与 MTE2→S 显式同步，语义等价（见 PLAN.md 决策记录）。
template <typename T>
__aicore__ inline static float ScalarToFloat(const GlobalTensor<T>& gm)
{
    const T v = gm.GetValue(0);
    if constexpr (Std::is_same<T, float>::value) {
        return v;
    } else if constexpr (Std::is_same<T, bfloat16_t>::value) {
        return AscendC::ToFloat(v); // 设备后端不支持 static_cast<float>(bfloat16_t)
    } else {
        return static_cast<float>(v); // half 的 C++ 显式转换
    }
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockFactor_ = (tilingData->blockFactor > 0) ? tilingData->blockFactor : 1;
    ubFactor_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : 1;
    x1InnerLen_ = tilingData->x1InnerLen;
    x1RowCount_ = tilingData->x1RowCount;
    x2InnerLen_ = tilingData->x2InnerLen;
    x2RowCount_ = tilingData->x2RowCount;
    x1Period_ = x1InnerLen_ * x1RowCount_;
    x2Period_ = x2InnerLen_ * x2RowCount_;
    x1Scalar_ = (x1Period_ == 1);
    x2Scalar_ = (x2Period_ == 1);

    const int64_t blockIdx = GetBlockIdx();
    blockStart_ = blockIdx * blockFactor_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x1, totalNum_);
    inputGMX2.SetGlobalBuffer((__gm__ T*)x2, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);

    // 标量值在 Init 一次性读取（S 管道 GM 读；标量在整个 kernel 生命周期不变）
    if (x1Scalar_) {
        x1ScalarValue_ = ScalarToFloat(inputGMX);
    }
    if (x2Scalar_) {
        x2ScalarValue_ = ScalarToFloat(inputGMX2);
    }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubFactor_ * sizeof(T));
    // UB 预算（DESIGN.md §1.5）：mode2（T=float）直算链只用 qF 一个 FP32 工作缓冲
    // （x1/x2 走 VECIN 队列），若也分配 x1F/x2F 会超 28B/元素预算（36B/元素）导致 UB 溢出。
    if constexpr (Std::is_same<T, float>::value) {
        pipe.InitBuffer(qFBuf, ubFactor_ * sizeof(float));
    } else {
        // mode0/1（T=half/bfloat16_t）升 FP32 计算：x1F/x2F/qF 三个 FP32 工作缓冲
        pipe.InitBuffer(x1FBuf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(x2FBuf, ubFactor_ * sizeof(float));
        pipe.InitBuffer(qFBuf, ubFactor_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Process()
{
    // 越界防护（DESIGN.md §3.5）：blockNum = ceil(totalNum / blockFactor) 可能小于实际核数
    if (blockStart_ >= totalNum_) {
        return;
    }
    const int64_t blockLen =
        (blockStart_ + blockFactor_ <= totalNum_) ? blockFactor_ : (totalNum_ - blockStart_);

    int64_t progress = 0;
    while (progress < blockLen) {
        const int64_t currentNum = (blockLen - progress < ubFactor_) ? (blockLen - progress) : ubFactor_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        progress += currentNum;
    }
}

// 非标量输入 tile 加载（DESIGN.md §2.4.3 策略 1/2/3；策略 0 由 CopyIn 单独处理）
template <typename T>
__aicore__ inline void TruncateMod<T>::LoadTile(const GlobalTensor<T>& gm, const LocalTensor<T>& local, int64_t s,
    int64_t tileLen, int64_t innerLen, int64_t rowCount)
{
    const int64_t period = innerLen * rowCount;
    const int64_t maskCap = 256 / static_cast<int64_t>(sizeof(T));

    if (s % innerLen + tileLen <= innerLen) {
        // 策略 1：tile 位于单个源行内，一次连续搬运
        const int64_t srcOff = ((s / innerLen) % rowCount) * innerLen + (s % innerLen);
        DataCopyPad<T>(local, gm[srcOff],
            DataCopyExtParams{1, static_cast<uint32_t>(tileLen * sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{});
    } else if (s % period == 0 && tileLen % period == 0 && (period * sizeof(T)) % 32 == 0 &&
               period <= maskCap) {
        // 策略 2：整周期重复。先搬入一个周期 [0,P)，再 mask 形式 Copy 扩展到 [P,T)。
        // 注意：period > maskCap 时掩码被钳制（掩码连续模式单次迭代上限 128/64 元素），
        // 必须退化为策略 3 —— 见 PLAN.md 决策记录（DESIGN.md 触发条件未含该上限的修正）。
        DataCopyPad<T>(local, gm[0],
            DataCopyExtParams{1, static_cast<uint32_t>(period * sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{});
        const int64_t k = tileLen / period;              // 重复倍数
        const uint16_t repBlocks = static_cast<uint16_t>(period * sizeof(T) / 32);
        for (int64_t off = 0; off < k - 1; off += 255) { // repeatTime 为 uint8_t，>255 分片
            const uint8_t n = static_cast<uint8_t>((k - 1 - off < 255) ? (k - 1 - off) : 255);
            Copy<T>(local[(off + 1) * period], local[0], static_cast<uint64_t>(period), n,
                CopyRepeatParams{1, 1, repBlocks, 0});
        }
    } else {
        // 策略 3：跨周期逐段兜底。tile 内按 period 边界切成连续段 [a,b)，每段源地址连续。
        int64_t pos = s;
        int64_t segOff = 0;
        const int64_t end = s + tileLen;
        while (pos < end) {
            int64_t nextB = ((pos / period) + 1) * period;
            if (nextB > end) {
                nextB = end;
            }
            const int64_t segLen = nextB - pos;
            const int64_t srcOff = ((pos / innerLen) % rowCount) * innerLen + (pos % innerLen);
            DataCopyPad<T>(local[segOff], gm[srcOff],
                DataCopyExtParams{1, static_cast<uint32_t>(segLen * sizeof(T)), 0, 0, 0},
                DataCopyPadExtParams<T>{});
            pos = nextB;
            segOff += segLen;
        }
    }
}

// 标量输入展开（DESIGN.md §2.4.2 策略 0）：
//  - mode0/1（T != float）：Duplicate 写 FP32 工作缓冲 fpBuf，Compute 跳过对应 Cast、不 DeQue；
//  - mode2（T == float）：Duplicate 写输入队列缓冲，Compute 照常 DeQue（V 管道顺序保证可见性）。
template <typename T>
__aicore__ inline void TruncateMod<T>::ExpandScalar(const GlobalTensor<T>& gm, float scalarValue,
    TQue<QuePosition::VECIN, BUFFER_NUM>& queue, TBuf<QuePosition::VECCALC>& fpBuf, int64_t currentNum, bool& enqueued)
{
    enqueued = false;
    if constexpr (Std::is_same<T, float>::value) {
        LocalTensor<T> local = queue.AllocTensor<T>();
        Duplicate<T>(local, static_cast<T>(scalarValue), static_cast<int32_t>(currentNum));
        queue.EnQue(local);
        enqueued = true;
    } else {
        LocalTensor<float> fp = fpBuf.Get<float>();
        Duplicate<float>(fp, scalarValue, static_cast<int32_t>(currentNum));
    }
    (void)gm; // 标量值已在 Init 通过 GetValue 读取，此处不再访问 GM
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    const int64_t s = blockStart_ + progress;

    if (!x1Scalar_) {
        LocalTensor<T> x1Local = inputQueueX.AllocTensor<T>();
        LoadTile(inputGMX, x1Local, s, currentNum, x1InnerLen_, x1RowCount_);
        inputQueueX.EnQue(x1Local);
    } else {
        bool enqueued = false;
        ExpandScalar(inputGMX, x1ScalarValue_, inputQueueX, x1FBuf, currentNum, enqueued);
    }

    if (!x2Scalar_) {
        LocalTensor<T> x2Local = inputQueueX2.AllocTensor<T>();
        LoadTile(inputGMX2, x2Local, s, currentNum, x2InnerLen_, x2RowCount_);
        inputQueueX2.EnQue(x2Local);
    } else {
        bool enqueued = false;
        ExpandScalar(inputGMX2, x2ScalarValue_, inputQueueX2, x2FBuf, currentNum, enqueued);
    }
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Compute(int64_t currentNum)
{
    const int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (Std::is_same<T, float>::value) {
        // mode2：FP32 直算。x1Local/x2Local 为 VECIN 队列缓冲，yLocal 为 VECOUT 输出缓冲。
        LocalTensor<T> x1Local = inputQueueX.DeQue<T>();
        LocalTensor<T> x2Local = inputQueueX2.DeQue<T>();
        LocalTensor<float> qF = qFBuf.Get<float>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

        Div(qF, x1Local, x2Local, count);                       // 商
        Cast<float, float>(qF, qF, RoundMode::CAST_TRUNC, count); // 向 0 截断
        Mul(qF, qF, x2Local, count);                            // q * x2
        Sub(yLocal, x1Local, qF, count);                        // y = x1 - q*x2

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(x1Local);
        inputQueueX2.FreeTensor(x2Local);
    } else {
        // mode0/1：半精度升 FP32 计算。x1F/x2F 为 FP32 工作缓冲；标量输入已在 CopyIn 由
        // Duplicate 写满对应 xF，此处跳过 Cast 且不 DeQue。
        LocalTensor<float> x1F = x1FBuf.Get<float>();
        LocalTensor<float> x2F = x2FBuf.Get<float>();
        if (!x1Scalar_) {
            LocalTensor<T> x1L = inputQueueX.DeQue<T>();
            Cast<float, T>(x1F, x1L, RoundMode::CAST_NONE, count);
            inputQueueX.FreeTensor(x1L);
        }
        if (!x2Scalar_) {
            LocalTensor<T> x2L = inputQueueX2.DeQue<T>();
            Cast<float, T>(x2F, x2L, RoundMode::CAST_NONE, count);
            inputQueueX2.FreeTensor(x2L);
        }

        LocalTensor<float> qF = qFBuf.Get<float>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

        Div(qF, x1F, x2F, count);                               // 商
        Cast<float, float>(qF, qF, RoundMode::CAST_TRUNC, count); // 向 0 截断
        Mul(qF, qF, x2F, count);                                // q * x2
        Sub(x1F, x1F, qF, count);                               // x1 = x1 - q*x2
        Cast<T, float>(yLocal, x1F, RoundMode::CAST_RINT, count); // 回写输出 dtype

        outputQueueY.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyPad<T>(outputGMY[blockStart_ + progress], yLocal,
        DataCopyExtParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0});
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsTruncateMod
#endif // TRUNCATEMOD_H
