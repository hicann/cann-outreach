// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

// 每个核单次搬运/计算的元素数
constexpr uint32_t TILE_LEN = 4096;
// 队列缓冲深度: 3级流水，搬运与计算重叠
constexpr uint32_t BUFFER_NUM = 3;
// Erf 接口共享临时缓冲区（字节），取足够大以覆盖接口需求
constexpr uint32_t ERF_TMP_BYTES = TILE_LEN * 4 * 4; // 64KB

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        // 每核处理的元素数，向上对齐到32元素，保证每核起始地址32字节对齐
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        perCore = (perCore + 31) / 32 * 32;
        uint32_t start = blockIdx * perCore;
        if (start >= length) {
            start = length;
            count = 0;
        } else {
            count = (length - start) < perCore ? (length - start) : perCore;
        }
        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + start, count);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + start, count);
        pipe.InitBuffer(inQue, BUFFER_NUM, TILE_LEN * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQue, BUFFER_NUM, TILE_LEN * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuf, ERF_TMP_BYTES);
    }

    // 经典三段流水: CopyIn -> Compute -> CopyOut，多级缓冲重叠搬运与计算
    __aicore__ inline void Process() {
        if (count == 0) {
            return;
        }
        uint32_t tiles = (count + TILE_LEN - 1) / TILE_LEN;
        uint32_t pre = tiles < (BUFFER_NUM - 1) ? tiles : (BUFFER_NUM - 1);
        for (uint32_t i = 0; i < pre; i++) {
            CopyIn(i);
        }
        for (uint32_t i = 0; i < tiles; i++) {
            Compute(i);
            if (i + pre < tiles) {
                CopyIn(i + pre);
            }
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t TileLen(uint32_t i) const {
        uint32_t off = i * TILE_LEN;
        uint32_t remain = count - off;
        return remain < TILE_LEN ? remain : TILE_LEN;
    }

    __aicore__ inline void CopyIn(uint32_t i) {
        LocalTensor<DT_INPUT_X> x = inQue.AllocTensor<DT_INPUT_X>();
        uint32_t bytes = static_cast<uint32_t>(TileLen(i) * sizeof(DT_INPUT_X));
        if ((bytes & 0x1F) == 0) {
            // 32字节对齐场景走普通DataCopy，开销更小
            DataCopy(x, xGm[i * TILE_LEN], TileLen(i));
        } else {
            // DataCopyPad 支持非32字节整数倍的搬运长度，兼容非对齐场景
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
            DataCopyPad(x, xGm[i * TILE_LEN], copyParams, padParams);
        }
        inQue.EnQue(x);
    }

    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    __aicore__ inline void Compute(uint32_t i) {
        uint32_t len = TileLen(i);
        LocalTensor<DT_INPUT_X> x = inQue.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> y = outQue.AllocTensor<DT_INPUT_X>();
        LocalTensor<uint8_t> sharedTmp = tmpBuf.Get<uint8_t>();

        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            // float16 原生半精度计算（910B 的 Erf 直接支持 half），省掉 Cast 开销
            ComputeHalf(x, y, sharedTmp, len);
        } else {
            LocalTensor<float> xf = x.template ReinterpretCast<float>();
            LocalTensor<float> yf = y.template ReinterpretCast<float>();
            if (len <= 1024) {
                // 小块: 指令调用次数主导开销，走调用次数最少的 Erf 路径
                ComputeFloatErf(xf, yf, sharedTmp, len);
            } else {
                // 大块: 手写轻量erf，吞吐更高
                LocalTensor<float> scratch = tmpBuf.Get<float>();
                ComputeFloat(xf, yf, scratch, len);
            }
        }

        inQue.FreeTensor(x);
        outQue.EnQue(y);
    }

    __aicore__ inline void ComputeHalf(const LocalTensor<half> &x, const LocalTensor<half> &y,
                                       const LocalTensor<uint8_t> &sharedTmp, uint32_t len) {
        const half RSQRT2 = static_cast<half>(0.70710678118654752440f); // 1/sqrt(2)
        const half ONE = static_cast<half>(1.0f);
        const half HALF = static_cast<half>(0.5f);
        Muls(y, x, RSQRT2, len);        // t = x / sqrt(2)
        Erf(y, y, sharedTmp, len);      // t = erf(t)
        Adds(y, y, ONE, len);           // t = 1 + erf(x/sqrt(2))
        Mul(y, y, x, len);              // t = x * (1 + erf(...))
        Muls(y, y, HALF, len);          // t = 0.5 * x * (1 + erf(...))
    }

    // fp32小块路径: Erf高阶API，调用次数最少
    __aicore__ inline void ComputeFloatErf(const LocalTensor<float> &xf, const LocalTensor<float> &yf,
                                           const LocalTensor<uint8_t> &sharedTmp, uint32_t len) {
        constexpr float RSQRT2 = 0.70710678118654752440f; // 1/sqrt(2)
        Muls(yf, xf, RSQRT2, len);      // t = x / sqrt(2)
        Erf(yf, yf, sharedTmp, len);    // t = erf(t)
        Adds(yf, yf, 1.0f, len);        // t = 1 + erf(x/sqrt(2))
        Mul(yf, yf, xf, len);           // t = x * (1 + erf(...))
        Muls(yf, yf, 0.5f, len);        // t = 0.5 * x * (1 + erf(...))
    }

    // fp32路径: 手写轻量erf, 避免Erf高阶API的固定开销
    // 利用 x*erf(x/sqrt(2)) = |x|*erf(|x|/sqrt(2)), 得 GELU(x) = 0.5*(x + |x| - |x|*P(t)*exp(-x^2/2))
    // 其中 t = 1/(1+0.2316419*|x|), P 为 Abramowitz-Stegun 7.1.26 多项式, erf绝对误差 <= 1.5e-7
    __aicore__ inline void ComputeFloat(const LocalTensor<float> &xf, const LocalTensor<float> &yf,
                                        const LocalTensor<float> &scratch, uint32_t len) {
        LocalTensor<float> s = scratch;                 // s = |x|
        LocalTensor<float> t = scratch[TILE_LEN];       // t = 1/(1+p*|x|)
        LocalTensor<float> poly = scratch[TILE_LEN * 2];
        LocalTensor<float> e = scratch[TILE_LEN * 3];   // e = exp(-x^2/2)

        Abs(s, xf, len);
        Muls(t, s, 0.2316419f, len);        // p' = 0.3275911/sqrt(2)
        Adds(t, t, 1.0f, len);
        Duplicate(e, 1.0f, len);
        Div(t, e, t, len);                  // t = 1/(1+p'*|x|)
        // Horner: poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        Muls(poly, t, 1.061405429f, len);
        Adds(poly, poly, -1.453152027f, len);
        Mul(poly, poly, t, len);
        Adds(poly, poly, 1.421413741f, len);
        Mul(poly, poly, t, len);
        Adds(poly, poly, -0.284496736f, len);
        Mul(poly, poly, t, len);
        Adds(poly, poly, 0.254829592f, len);
        Mul(poly, poly, t, len);
        // e = exp(-x^2/2)
        Mul(e, s, s, len);
        Muls(e, e, -0.5f, len);
        Exp(e, e, len);
        // y = 0.5*(x + s - s*poly*e)
        Mul(poly, poly, e, len);
        Mul(poly, poly, s, len);
        Add(yf, xf, s, len);
        Sub(yf, yf, poly, len);
        Muls(yf, yf, 0.5f, len);
    }

    __aicore__ inline void CopyOut(uint32_t i) {
        LocalTensor<DT_INPUT_X> y = outQue.DeQue<DT_INPUT_X>();
        uint32_t bytes = static_cast<uint32_t>(TileLen(i) * sizeof(DT_INPUT_X));
        if ((bytes & 0x1F) == 0) {
            DataCopy(yGm[i * TILE_LEN], y, TileLen(i));
        } else {
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPad(yGm[i * TILE_LEN], y, copyParams);
        }
        outQue.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<DT_INPUT_X> xGm;
    GlobalTensor<DT_INPUT_X> yGm;
    uint32_t count = 0;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}
