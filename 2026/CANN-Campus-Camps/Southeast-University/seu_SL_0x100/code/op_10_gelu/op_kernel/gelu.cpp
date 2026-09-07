// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;                     // 输入/输出队列双缓冲
constexpr float GELU_INV_SQRT2 = 0.70710678118654752f; // 1 / sqrt(2)
constexpr float GELU_HALF = 0.5f;
} // namespace

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData &tilingData) {
        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * tilingData.blockLength;
        uint64_t end = start + tilingData.blockLength;
        end = end < tilingData.totalLength ? end : tilingData.totalLength;
        myLength_ = start < end ? static_cast<uint32_t>(end - start) : 0;
        tileLength_ = tilingData.tileLength;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + start, myLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output) + start, myLength_);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(absBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(erfInBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(erfOutBuf_, tileLength_ * sizeof(float));
        // Erf 高阶 API 需要的临时 buffer: 每个元素 3 个 float (ERF_FLOAT_CALC_PROCEDURE)
        pipe_.InitBuffer(erfTmpBuf_, 3 * tileLength_ * sizeof(float));
        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            // fp16 升精度到 fp32 计算
            pipe_.InitBuffer(xFp32Buf_, tileLength_ * sizeof(float));
            pipe_.InitBuffer(yFp32Buf_, tileLength_ * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        uint32_t tileNum = (myLength_ + tileLength_ - 1) / tileLength_;
        for (uint32_t i = 0; i < tileNum; ++i) {
            uint32_t remain = myLength_ - i * tileLength_;
            uint32_t len = remain < tileLength_ ? remain : tileLength_;
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len) {
        LocalTensor<DT_INPUT_X> xLocal = inQueueX_.AllocTensor<DT_INPUT_X>();
        // 非对齐场景使用 DataCopyPad, 仅搬运 len 个有效元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[progress * tileLength_], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_INPUT_X> xLocal = inQueueX_.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> yLocal = outQueueY_.AllocTensor<DT_INPUT_X>();
        LocalTensor<float> absLocal = absBuf_.Get<float>();
        LocalTensor<float> erfInLocal = erfInBuf_.Get<float>();
        LocalTensor<float> erfOutLocal = erfOutBuf_.Get<float>();
        LocalTensor<uint8_t> erfTmpLocal = erfTmpBuf_.Get<uint8_t>();
        LocalTensor<float> xFp32;
        LocalTensor<float> yFp32;
        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            xFp32 = xFp32Buf_.Get<float>();
            yFp32 = yFp32Buf_.Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, len);
        } else {
            xFp32 = xLocal;
            yFp32 = yLocal;
        }
        // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // 利用 x * erf(x / sqrt(2)) = |x| * erf(|x| / sqrt(2)), 转化为无符号分支的计算:
        // gelu(x) = 0.5 * x + 0.5 * |x| * erf(|x| / sqrt(2))
        Abs(absLocal, xFp32, len);
        Muls(erfInLocal, absLocal, GELU_INV_SQRT2, len);
        // Erf 的 src/dst/tmp 不允许地址重叠
        Erf(erfOutLocal, erfInLocal, erfTmpLocal, len);
        Mul(erfOutLocal, absLocal, erfOutLocal, len);
        Muls(erfOutLocal, erfOutLocal, GELU_HALF, len);
        Muls(yFp32, xFp32, GELU_HALF, len);
        Add(yFp32, yFp32, erfOutLocal, len);
        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            Cast(yLocal, yFp32, RoundMode::CAST_RINT, len);
        }
        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len) {
        LocalTensor<DT_INPUT_X> yLocal = outQueueY_.DeQue<DT_INPUT_X>();
        // 仅写回 len 个有效元素, 非对齐部分由 DataCopyPad 处理
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPad(yGm_[progress * tileLength_], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> erfInBuf_;
    TBuf<TPosition::VECCALC> erfOutBuf_;
    TBuf<TPosition::VECCALC> erfTmpBuf_;
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> yFp32Buf_;
    GlobalTensor<DT_INPUT_X> xGm_;
    GlobalTensor<DT_INPUT_X> yGm_;
    uint32_t myLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    if ASCEND_IS_AIC {
        return;
    }
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
