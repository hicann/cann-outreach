// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// GELU(x) = x * Phi(x)，Phi 为标准正态分布累积分布函数。
// 使用 Abramowitz-Stegun 7.1.26 对 Phi 的高精度近似（绝对误差 < 7.5e-8），
// 并化为符号无关的形式：
//   GELU(x) = 0.5 * (x + |x|) - |x| * Q(|x|)
//   Q(a)    = phi(a) * poly(t)
//   t       = 1 / (1 + 0.2316419 * a)
//   phi(a)  = (1 / sqrt(2*pi)) * exp(-0.5 * a * a)
constexpr float GELU_P = 0.2316419f;
constexpr float GELU_B1 = 0.319381530f;
constexpr float GELU_B2 = -0.356563782f;
constexpr float GELU_B3 = 1.781477937f;
constexpr float GELU_B4 = -1.821255978f;
constexpr float GELU_B5 = 1.330274429f;
constexpr float GELU_C = 0.3989422804014327f;

template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                uint32_t totalLength, uint32_t tileLength, uint32_t alignNum) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->alignNum = alignNum;
        this->totalTiles = (totalLength + tileLength - 1) / tileLength;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t perCoreTiles = (this->totalTiles + blockNum - 1) / blockNum;
        this->tileStart = blockIdx * perCoreTiles;
        this->tileEnd = this->tileStart + perCoreTiles;
        if (this->tileEnd > this->totalTiles) {
            this->tileEnd = this->totalTiles;
        }

        xGm.SetGlobalBuffer((__gm__ T *)input_x, totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)output, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));

        // float 中间缓冲区
        if constexpr (sizeof(T) == 2) {
            pipe.InitBuffer(tmpBufX, tileLength * sizeof(float));
            pipe.InitBuffer(tmpBufY, tileLength * sizeof(float));
        }
        pipe.InitBuffer(tmpBufA, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBufS1, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBufS2, tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t tile = this->tileStart; tile < this->tileEnd; tile++) {
            uint32_t offset = tile * this->tileLength;
            uint32_t len = this->totalLength - offset;
            if (len > this->tileLength) {
                len = this->tileLength;
            }
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (len % this->alignNum == 0) {
            DataCopy(xLocal, xGm[offset], len);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

   __aicore__ inline void Compute(uint32_t len)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    if constexpr (sizeof(T) == 2) {
        LocalTensor<float> xBuf = tmpBufX.Get<float>();
        LocalTensor<float> yBuf = tmpBufY.Get<float>();

        Cast(xBuf, xLocal, RoundMode::CAST_NONE, len);

        GeluCompute(xBuf, yBuf, len);

        Cast(yLocal, yBuf, RoundMode::CAST_ROUND, len);
    } else {
        GeluCompute(xLocal, yLocal, len);
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

__aicore__ inline void GeluCompute(LocalTensor<float> &x,
                                   LocalTensor<float> &y,
                                   uint32_t count)
{
    LocalTensor<float> s1 = tmpBufS1.Get<float>();

    constexpr float INV_SQRT2 = 0.7071067811865475244f;

    // s1 = x / sqrt(2)
    Muls(s1, x, INV_SQRT2, count);

    // s1 = erf(x / sqrt(2))
    Erf(s1, s1, count);

    // s1 = 1 + erf(...)
    Adds(s1, s1, 1.0f, count);

    // s1 = 0.5 * (1 + erf(...))
    Muls(s1, s1, 0.5f, count);

    // y = x * s1
    Mul(y, x, s1, count);
}

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (len % this->alignNum == 0) {
            DataCopy(yGm[offset], yLocal, len);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBufX;
    TBuf<QuePosition::VECCALC> tmpBufY;
    TBuf<QuePosition::VECCALC> tmpBufA;
    TBuf<QuePosition::VECCALC> tmpBufS1;
    TBuf<QuePosition::VECCALC> tmpBufS2;

    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t alignNum;
    uint32_t totalTiles;
    uint32_t tileStart;
    uint32_t tileEnd;

};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.totalLength, tiling_data.tileLength, tiling_data.alignNum);
    op.Process();
}