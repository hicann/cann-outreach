// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

// 每个核单次搬运/计算的元素数
constexpr uint32_t TILE_LEN = 4096;
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
        pipe.InitBuffer(inQue, 1, TILE_LEN * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQue, 1, TILE_LEN * sizeof(DT_INPUT_X));
        pipe.InitBuffer(xBuf, TILE_LEN * sizeof(float));
        pipe.InitBuffer(yBuf, TILE_LEN * sizeof(float));
        pipe.InitBuffer(tmpBuf, ERF_TMP_BYTES);
    }

    __aicore__ inline void Process() {
        for (uint32_t off = 0; off < count; off += TILE_LEN) {
            uint32_t tileLen = (count - off) < TILE_LEN ? (count - off) : TILE_LEN;
            CopyIn(off, tileLen);
            Compute(tileLen);
            CopyOut(off, tileLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t off, uint32_t len) {
        LocalTensor<DT_INPUT_X> x = inQue.AllocTensor<DT_INPUT_X>();
        // DataCopyPad 支持非32字节整数倍的搬运长度，兼容非对齐场景
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[off], copyParams, padParams);
        inQue.EnQue(x);
    }

    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<DT_INPUT_X> x = inQue.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> y = outQue.AllocTensor<DT_INPUT_X>();
        LocalTensor<uint8_t> sharedTmp = tmpBuf.Get<uint8_t>();

        if constexpr (IsSameType<DT_INPUT_X, half>::value) {
            // float16 先转 float32 计算，保证精度
            LocalTensor<float> xf = xBuf.Get<float>();
            LocalTensor<float> yf = yBuf.Get<float>();
            Cast(xf, x, RoundMode::CAST_NONE, len);
            ComputeFloat(xf, yf, sharedTmp, len);
            Cast(y, yf, RoundMode::CAST_NONE, len);
        } else {
            LocalTensor<float> xf = x.template ReinterpretCast<float>();
            LocalTensor<float> yf = y.template ReinterpretCast<float>();
            ComputeFloat(xf, yf, sharedTmp, len);
        }

        inQue.FreeTensor(x);
        outQue.EnQue(y);
    }

    __aicore__ inline void ComputeFloat(const LocalTensor<float> &xf, const LocalTensor<float> &yf,
                                        const LocalTensor<uint8_t> &sharedTmp, uint32_t len) {
        constexpr float RSQRT2 = 0.70710678118654752440f; // 1/sqrt(2)
        Muls(yf, xf, RSQRT2, len);      // t = x / sqrt(2)
        Erf(yf, yf, sharedTmp, len);    // t = erf(t)
        Adds(yf, yf, 1.0f, len);        // t = 1 + erf(x/sqrt(2))
        Mul(yf, yf, xf, len);           // t = x * (1 + erf(...))
        Muls(yf, yf, 0.5f, len);        // t = 0.5 * x * (1 + erf(...))
    }

    __aicore__ inline void CopyOut(uint32_t off, uint32_t len) {
        LocalTensor<DT_INPUT_X> y = outQue.DeQue<DT_INPUT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopyPad(yGm[off], y, copyParams);
        outQue.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> yBuf;
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
