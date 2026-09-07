// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;                  // 队列缓冲个数，开启 double buffer 流水
constexpr uint32_t TILE_LENGTH = 2048;             // 每个计算分块的元素个数
constexpr float INV_SQRT2 = 0.70710678118654752f;  // 1 / sqrt(2)

// GELU 精确公式：y = 0.5 * x * (1 + erf(x / sqrt(2)))
// 说明：tanh 近似与精确 GELU 的最大绝对偏差约 5e-4，无法满足 float32 双万分之一(1e-4)的精度要求，
// 因此本实现采用 erf 精确公式。erf 通过 Abramowitz & Stegun (7.1.26) 多项式计算，
// 其对 erf 的逼近误差 <= 1.5e-7，叠加 fp32 运算舍入后 GELU 结果绝对误差 <= 5e-7，远优于 1e-4。
constexpr float ERF_P  = 0.3275911f;
constexpr float ERF_A1 = 0.254829592f;
constexpr float ERF_A2 = -0.284496736f;
constexpr float ERF_A3 = 1.421413741f;
constexpr float ERF_A4 = -1.453152027f;
constexpr float ERF_A5 = 1.061405429f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        // 多核均分：当前核处理区间 [blockOffset, blockOffset + blockLength)
        // 分核边界无需 32 字节对齐，搬入搬出由 DataCopyPad 处理非对齐场景
        uint32_t blockNum = GetBlockNum();
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        uint32_t start = blockNum > 0 ? GetBlockIdx() * perCore : 0;
        this->blockOffset = start;
        this->blockLength = (start >= length) ? 0 : ((length - start < perCore) ? (length - start) : perCore);

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x, length);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output, length);

        // 输入/输出队列（double buffer）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
        // 中间计算统一使用 fp32 缓冲：fp16 输入先升精度再计算，保证双千分之一精度
        pipe.InitBuffer(x32Buf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(absBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(zBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(polyBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(resBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 尾块分块循环：length 为任意正整数（含非 32 字节对齐长度），逐块处理
        for (uint32_t offset = 0; offset < this->blockLength; offset += TILE_LENGTH) {
            uint32_t curLen = (this->blockLength - offset < TILE_LENGTH) ?
                              (this->blockLength - offset) : TILE_LENGTH;
            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_INPUT_X> padParams{true, 0, 0, static_cast<DT_INPUT_X>(0)};
        DataCopyPad(xLocal, xGm[this->blockOffset + offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();

        LocalTensor<float> x32 = x32Buf.Get<float>();
        LocalTensor<float> absX = absBuf.Get<float>();
        LocalTensor<float> z = zBuf.Get<float>();
        LocalTensor<float> t = tBuf.Get<float>();
        LocalTensor<float> poly = polyBuf.Get<float>();
        LocalTensor<float> res = resBuf.Get<float>();

        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            Cast(x32, xLocal, RoundMode::CAST_NONE, length);
        } else {
            Adds(x32, xLocal, 0.0f, length);
        }

        // z = |x| / sqrt(2)
        Abs(absX, x32, length);
        Muls(z, absX, INV_SQRT2, length);

        
        // 此时 erf 趋近 1，结果自然趋于渐近值（x>0 输出 x，x<0 输出 0），数值安全
        Mul(res, z, z, length);
        Muls(res, res, -1.0f, length);
        Exp(res, res, length);

        // 4. t = 1 / (1 + p*z)
        Muls(t, z, ERF_P, length);
        Adds(t, t, 1.0f, length);
        Duplicate(poly, 1.0f, length);
        Div(t, poly, t, length);

        // 5. poly = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)，Horner 形式
        Muls(poly, t, ERF_A5, length);
        Adds(poly, poly, ERF_A4, length);
        Mul(poly, poly, t, length);
        Adds(poly, poly, ERF_A3, length);
        Mul(poly, poly, t, length);
        Adds(poly, poly, ERF_A2, length);
        Mul(poly, poly, t, length);
        Adds(poly, poly, ERF_A1, length);
        Mul(poly, poly, t, length);

        // 6. erf(z) = 1 - poly * exp(-z^2)
        Mul(poly, poly, res, length);
        Muls(poly, poly, -1.0f, length);
        Adds(poly, poly, 1.0f, length);

        // 7. y = 0.5 * (x + |x| * erf(|x| / sqrt(2)))
        //    利用 erf 奇函数性质：x*erf(x/sqrt(2)) == |x|*erf(|x|/sqrt(2))，无需符号判断
        Mul(res, absX, poly, length);
        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            Add(res, x32, res, length);
            Muls(res, res, 0.5f, length);
            // fp32 结果就近舍入（round to nearest even）回 fp16，保证输出精度
            Cast(yLocal, res, RoundMode::CAST_RINT, length);
        } else {
            Add(yLocal, x32, res, length);
            Muls(yLocal, yLocal, 0.5f, length);
        }

        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        // DataCopyPad 搬出同样支持非 32 字节对齐长度，仅写出 length 个有效元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPad(yGm[this->blockOffset + offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> x32Buf;
    TBuf<QuePosition::VECCALC> absBuf;
    TBuf<QuePosition::VECCALC> zBuf;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<QuePosition::VECCALC> polyBuf;
    TBuf<QuePosition::VECCALC> resBuf;
    GlobalTensor<DT_INPUT_X> xGm;
    GlobalTensor<DT_INPUT_X> yGm;
    uint32_t blockOffset;
    uint32_t blockLength;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}
