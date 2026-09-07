// Kernel侧核函数实现 —— y = GELU(x) = x * 0.5 * (1 + erf(x / √2))
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr uint32_t BUF_DEPTH = 2;      // v5 已验证最优
constexpr float RSQRT2 = 0.7071067811865476f;  // 1/√2

template <class DT_INPUT_X>
class KernelGelu {
    // 单队列直写结构（学流派 B）：砍掉 VECOUT 队列 —— 省每 tile 3 次同步（AllocTensor/EnQue/DeQue）
    // 计算 in-place 写回 xLocal（向量指令 dst==src 安全，Adds(e,e,1) 一直这么用），算完直接 MTE3 搬出
    static constexpr uint32_t TILE_LENGTH = 4096u;
    static constexpr uint32_t TILE_SHIFT  = 12u;
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData &tilingData) {
        // Pipe & Queue
        pipe_.InitBuffer(queueX_, BUF_DEPTH, sizeof(DT_INPUT_X) * TILE_LENGTH);
        pipe_.InitBuffer(queueY_, BUF_DEPTH, sizeof(DT_INPUT_X) * TILE_LENGTH);
        // t/e 中间量共一个 buffer 前后两半（half/fp32 按路径各用各的尺寸）
        pipe_.InitBuffer(bufTmp_, 2 * sizeof(DT_INPUT_X) * TILE_LENGTH);

        // GlobalTensor
        xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x);
        yGm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)output);

        uint32_t length = tilingData.length;

        // Tiling —— host 算好 perCore（已 32B 对齐），kernel 全程零软除法
        uint32_t alignElements = (uint32_t)(32 / sizeof(DT_INPUT_X));  // fp32=8, fp16=16（2 的幂）
        uint32_t perCore = tilingData.perCore;

        blockIdx_ = (uint32_t)AscendC::GetBlockIdx();

        coreOffset_ = perCore * blockIdx_;
        if (coreOffset_ >= length) {
            coreLen_ = 0;
        } else {
            uint32_t coreLenRaw = (coreOffset_ + perCore > length)
                                   ? (length - coreOffset_)
                                   : perCore;
            // CeilAlign 位运算（alignElements 是 2 的幂），杜绝软除法
            coreLen_ = (coreLenRaw + alignElements - 1) & ~(alignElements - 1);
        }

        // tileNum：CeilDiv 移位（TILE_LENGTH = 2^TILE_SHIFT）
        tileNum_ = (coreLen_ + TILE_LENGTH - 1) >> TILE_SHIFT;
    }

    __aicore__ inline void Process() {
        if (coreLen_ == 0) {
            return;
        }
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute();
            CopyOut();
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUF_DEPTH> queueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUF_DEPTH> queueY_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp_;

    AscendC::GlobalTensor<DT_INPUT_X> xGm_;
    AscendC::GlobalTensor<DT_INPUT_X> yGm_;

    uint32_t blockNum_   = 0;
    uint32_t blockIdx_   = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreLen_    = 0;
    uint32_t tileNum_    = 0;

    uint32_t currentOutIdx_  = 0;
    uint32_t currentRealLen_ = 0;

    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = coreOffset_ + tileIdx * TILE_LENGTH;
        // 最后一块可能不足 TILE_LENGTH（coreLen_ 与 TILE_LENGTH 都对齐，故 realLen 天然对齐）
        uint32_t realLen = (offset + TILE_LENGTH > coreOffset_ + coreLen_)
                           ? (coreOffset_ + coreLen_ - offset)
                           : TILE_LENGTH;
        currentRealLen_ = realLen;

        AscendC::LocalTensor<DT_INPUT_X> xLocal = queueX_.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, xGm_[offset], realLen);
        queueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = queueX_.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = queueY_.AllocTensor<DT_INPUT_X>();

        uint32_t n = currentRealLen_;

        // y = x * 0.5 * (1 + erf(x/√2))，共 5 个计算 API（fp32 最低步数）
        AscendC::LocalTensor<DT_INPUT_X> tmp = bufTmp_.Get<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> t = tmp;                 // 前半
        AscendC::LocalTensor<DT_INPUT_X> e = tmp[TILE_LENGTH];    // 后半

        if constexpr (sizeof(DT_INPUT_X) == 4) {
            // fp32 路径：Erf 精确版（fp32 要求 1e-4，tanh 近似误差 3e-4 过不了）
            AscendC::Muls(t, xLocal, RSQRT2, n);
            AscendC::Erf(e, t, n);                                    // e = erf(t)
            AscendC::Adds(e, e, static_cast<DT_INPUT_X>(1.0f), n);    // e = 1 + e
            AscendC::Muls(e, e, static_cast<DT_INPUT_X>(0.5f), n);    // e = 0.5 * (1 + e)
            AscendC::Mul(yLocal, xLocal, e, n);                       // y = x * 0.5 * (1 + erf(x/√2))
        } else {
            // fp16：half 直算 tanh（v5 已验证最优；cast-fp32 版实测 T5 18.24 已证伪）
            // GELU(x) ≈ 0.5x(1 + tanh(0.7978845608x + 0.0356774081x³))，误差 ~3e-4 << 1e-3
            AscendC::Mul(t, xLocal, xLocal, n);                       // t = x²
            AscendC::Muls(t, t, static_cast<DT_INPUT_X>(0.0356774081f), n);  // t = 0.0356774x²
            AscendC::Adds(t, t, static_cast<DT_INPUT_X>(0.7978845608f), n);  // t = 0.79788 + 0.0356774x²
            AscendC::Mul(t, t, xLocal, n);                            // t = 0.79788x + 0.0356774x³
            AscendC::Tanh(e, t, n);                                   // tanh(t)
            AscendC::Adds(e, e, static_cast<DT_INPUT_X>(1.0f), n);    // 1 + tanh
            AscendC::Mul(yLocal, xLocal, e, n);                       // x(1+tanh)
            AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.5f), n); // 0.5x(1+tanh)
        }

        queueY_.EnQue(yLocal);
        queueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = queueY_.DeQue<DT_INPUT_X>();

        uint32_t offset = coreOffset_ + currentOutIdx_ * TILE_LENGTH;
        AscendC::DataCopy(yGm_[offset], yLocal, currentRealLen_);

        queueY_.FreeTensor(yLocal);
        currentOutIdx_++;
    }
};

// ========== tiling key 模式分发由 tiling_key_gelu.h 的 ASCENDC_TPL_SEL 自动处理 ==========
template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
