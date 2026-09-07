#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

// 精确 GELU：GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// 等价变换：GELU(x) = 0.5 * (x + |x| * erf(|x| / sqrt(2)))
// 使用 Erf 精确计算，满足精度要求
constexpr float GELU_INV_SQRT2 = 0.7071067811865476f;
constexpr float GELU_HALF = 0.5f;

// ---- 将输入提升至 float 域（half -> float，float 直接复用） ----
__aicore__ inline AscendC::LocalTensor<float> GeluXF32(AscendC::LocalTensor<float> &xLocal,
                                                       AscendC::LocalTensor<float> &dst, uint32_t len)
{
    (void)dst; (void)len;
    return xLocal;
}
__aicore__ inline AscendC::LocalTensor<float> GeluXF32(AscendC::LocalTensor<half> &xLocal,
                                                       AscendC::LocalTensor<float> &dst, uint32_t len)
{
    AscendC::Cast(dst, xLocal, AscendC::RoundMode::CAST_NONE, len);
    return dst;
}

// ---- 写回输出（float 直接乘 0.5，half 先乘再转回） ----
__aicore__ inline void GeluStore(AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<float> &acc, uint32_t len)
{
    AscendC::Muls(yLocal, acc, GELU_HALF, len);
}
__aicore__ inline void GeluStore(AscendC::LocalTensor<half> &yLocal, AscendC::LocalTensor<float> &acc, uint32_t len)
{
    AscendC::Muls(acc, acc, GELU_HALF, len);
    AscendC::Cast(yLocal, acc, AscendC::RoundMode::CAST_RINT, len);
}

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData *tiling)
    {
        totalLength_ = tiling->totalLength;
        blockLength_ = tiling->blockLength;
        tileLength_ = tiling->tileLength;
        if (tileLength_ < 1) tileLength_ = 1;

        inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(input_x), totalLength_);
        outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), totalLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        // float 工作区：A=输入（half 时用），B=中间，C=累加
        pipe_.InitBuffer(tmpBufA_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufB_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufC_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIndex = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(blockLength_);
        if (startIndex >= static_cast<int64_t>(totalLength_)) return;

        int64_t curLen = static_cast<int64_t>(blockLength_);
        if (startIndex + curLen > static_cast<int64_t>(totalLength_))
            curLen = static_cast<int64_t>(totalLength_) - startIndex;
        if (curLen <= 0) return;

        int32_t loopCount = static_cast<int32_t>((curLen + static_cast<int64_t>(tileLength_) - 1) / static_cast<int64_t>(tileLength_));
        for (int32_t i = 0; i < loopCount; ++i) {
            int64_t offset = startIndex + static_cast<int64_t>(i) * static_cast<int64_t>(tileLength_);
            int64_t remain = curLen - static_cast<int64_t>(i) * static_cast<int64_t>(tileLength_);
            uint32_t len = tileLength_;
            if (static_cast<int64_t>(len) > remain) len = static_cast<uint32_t>(remain);

            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX_.AllocTensor<DT_INPUT_X>();
        // 使用 DataCopyPad 自动处理非对齐
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, inputGMX_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX_.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY_.AllocTensor<DT_INPUT_X>();
        AscendC::LocalTensor<float> tmpA = tmpBufA_.Get<float>();
        AscendC::LocalTensor<float> tmpB = tmpBufB_.Get<float>();
        AscendC::LocalTensor<float> tmpC = tmpBufC_.Get<float>();

        // 1. 将输入转为 float（half 时提升）
        AscendC::LocalTensor<float> xF32 = GeluXF32(xLocal, tmpA, len);

        // 2. 计算 |x| 和 |x|/sqrt(2)
        AscendC::Abs(tmpB, xF32, len);                     // B = |x|
        AscendC::Muls(tmpC, tmpB, GELU_INV_SQRT2, len);    // C = |x|/sqrt(2)

        // 3. erf(C)
        AscendC::Erf(tmpB, tmpC, len);                     // B = erf(C)

        // 4. |x| * erf(|x|/sqrt(2))
        AscendC::Abs(tmpC, xF32, len);                     // C = |x|（重算一次，避免额外 buffer）
        AscendC::Mul(tmpC, tmpC, tmpB, len);               // C = |x| * erf(...)

        // 5. 累加 x + |x|*erf(...)，再乘 0.5
        AscendC::Add(tmpC, tmpC, xF32, len);               // C = x + |x|*erf(...)
        GeluStore(yLocal, tmpC, len);                      // y = 0.5 * C

        inQueueX_.FreeTensor(xLocal);
        outQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY_.DeQue<DT_INPUT_X>();
        // 只拷贝有效数据（DataCopyPad 3 参数版本，不填充）
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufA_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufB_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufC_;
    AscendC::GlobalTensor<DT_INPUT_X> inputGMX_;
    AscendC::GlobalTensor<DT_INPUT_X> outputGMY_;
    uint32_t totalLength_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, &tiling_data);
    op.Process();
}