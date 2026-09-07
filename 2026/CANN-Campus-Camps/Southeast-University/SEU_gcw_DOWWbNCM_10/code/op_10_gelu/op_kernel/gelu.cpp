// Kernel侧核函数实现
#include "kernel_operator.h"

#include <type_traits>

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// 等价变形（无需判断符号，省掉分支/掩码）：
//   GELU(x) = 0.5 * (x + |x| * erf(|x| / sqrt(2)))
// 注意：必须走 erf 精确式。原生 AscendC::Gelu 是 sigmoid/tanh 近似，
// 最大绝对误差 4.7e-4，不满足本题 fp32 绝对误差 < 1e-4 的要求。
constexpr float GELU_INV_SQRT2 = 0.7071067811865476f;
constexpr float GELU_HALF = 0.5f;

// ---- dtype 自适应计算路径 ----
// fp16 / fp32 都走 erf 精确式。
// 原生 AscendC::Gelu 是 sigmoid/tanh 近似，绝对误差 ~4.7e-4；在 [-1,1] 小值区间相对误差可达 0.4+，
// 不满足 fp16 相对误差 < 1e-3 的要求，因此 fp16 也不能用原生 Gelu 加速。
// fp16 输入先 Cast 到 float 域再算 erf（fp16 域直接算 erf 误差 1.5e-3，超 1e-3）。

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData *tiling)
    {
        totalLength_ = tiling->totalLength;
        blockLength_ = tiling->blockLength;
        tileLength_ = tiling->tileLength;
        if (tileLength_ < 1) {
            tileLength_ = 1;
        }

        inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(input_x), totalLength_);
        outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), totalLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        // float 域工作区：A=输入 x(仅 half 用到)  B=ax 与 e  C=u / t / 累加
        pipe_.InitBuffer(tmpBufA_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufB_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufC_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIndex = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(blockLength_);
        if (startIndex >= static_cast<int64_t>(totalLength_)) {
            return;  // 空核直接退出
        }
        int64_t curLen = static_cast<int64_t>(blockLength_);
        if (startIndex + curLen > static_cast<int64_t>(totalLength_)) {
            curLen = static_cast<int64_t>(totalLength_) - startIndex;  // 尾核变短
        }
        if (curLen <= 0) {
            return;
        }

        int32_t loopCount = static_cast<int32_t>((curLen + static_cast<int64_t>(tileLength_) - 1) /
                                                 static_cast<int64_t>(tileLength_));
        for (int32_t i = 0; i < loopCount; ++i) {
            int64_t offset = startIndex + static_cast<int64_t>(i) * static_cast<int64_t>(tileLength_);
            int64_t remain = curLen - static_cast<int64_t>(i) * static_cast<int64_t>(tileLength_);
            uint32_t len = tileLength_;
            if (static_cast<int64_t>(len) > remain) {
                len = static_cast<uint32_t>(remain);  // 尾块截断，不越界
            }
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX_.AllocTensor<DT_INPUT_X>();
        // 非 32 字节对齐场景：用 DataCopyPad（blockLen 单位 Byte），不用 DataCopy
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

        // 取 float 域的 x：fp32 零拷贝复用 VECIN；fp16 提精度到 float（fp16 域直接算 erf 误差达 1.5e-3，超 1e-3）
        AscendC::LocalTensor<float> xF32;
        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            AscendC::Cast(tmpA, xLocal, AscendC::RoundMode::CAST_NONE, len);
            xF32 = tmpA;
        } else {
            xF32 = xLocal;  // 零拷贝
        }

        // 等价变形 GELU(x)=0.5*(x+|x|*erf(|x|/sqrt(2)))，免符号分支/掩码。
        // 两种 dtype 都走 erf 精确式：原生 AscendC::Gelu 近似误差 ~4.7e-4，但在 [-1,1] 区间
        // 相对误差可达 0.4+，不满足 fp16 相对误差 < 1e-3，故不能用于 fp16 加速。
        AscendC::Abs(tmpB, xF32, len);                  // B = |x|
        AscendC::Muls(tmpC, tmpB, GELU_INV_SQRT2, len); // C = |x| / sqrt(2)
        AscendC::Erf(tmpB, tmpC, len);                  // B = erf(C)（源目的不重叠）
        AscendC::Abs(tmpC, xF32, len);                  // C = |x|（复用，省一个 float buffer）
        AscendC::Mul(tmpC, tmpC, tmpB, len);            // C = |x| * erf(...)
        AscendC::Add(tmpC, tmpC, xF32, len);            // C = x + |x| * erf(...)

        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            AscendC::Muls(tmpC, tmpC, GELU_HALF, len);
            AscendC::Cast(yLocal, tmpC, AscendC::RoundMode::CAST_RINT, len);
        } else {
            AscendC::Muls(yLocal, tmpC, GELU_HALF, len);
        }

        inQueueX_.FreeTensor(xLocal);
        outQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY_.DeQue<DT_INPUT_X>();
        // UB -> GM：DataCopyPad 会丢弃补齐的 dummy 字节，不会越界写
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
