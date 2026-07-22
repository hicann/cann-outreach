#include <cmath>
#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t tileNum, uint32_t totalLength)
    {
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = totalLength;
        this->xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x);
        this->yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y);
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < static_cast<int32_t>(this->tileNum); ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline float TanhApprox(float x)
    {
        if (x > 6.0f) {
            return 1.0f;
        }
        if (x < -6.0f) {
            return -1.0f;
        }
        const float x2 = x * x;
        const float numerator = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        const float denominator = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
        return numerator / denominator;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        (void)progress;
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        const uint32_t tileIndex = static_cast<uint32_t>(progress);
        const uint32_t offset = tileIndex * this->blockLength;
        uint32_t validLength = this->blockLength;
        if (offset + validLength > this->tileLength) {
            validLength = this->tileLength - offset;
        }
        if (validLength == 0) {
            return;
        }

        for (uint32_t i = 0; i < validLength; ++i) {
            const float value = static_cast<float>(this->xGm.GetValue(offset + i));
            this->yGm.SetValue(offset + i, static_cast<DTYPE_Y>(TanhApprox(value)));
        }
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        (void)progress;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh kernel;
    kernel.Init(x, y, tilingData.blockLength, tilingData.tileNum, tilingData.tileLength);
    kernel.Process();
}
