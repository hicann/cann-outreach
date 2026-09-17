/**
 * @file inplace_rsqrt.cpp
 * @brief Ascend C InplaceRsqrt: self = 1 / sqrt(self)
 *        Shape: 4D [N4, N3, N2, N1] = [2, 2, 4, 128]
 *        Dtype: float16 (half), Format: ND
 *
 *        Core: AscendC::Rsqrt (vectorized vrsqrt), modifies LocalTensor in-place.
 */
#include "inplace_rsqrt.h"

constexpr int32_t TOTAL_LENGTH = 2 * 2 * 4 * 128; // 2048
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM; // 256
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // 64

class KernelInplaceRsqrt {
public:
    __aicore__ inline KernelInplaceRsqrt() {}

    __aicore__ inline void Init(GM_ADDR x)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(xBuf, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<half> xLocal = xBuf.Get<half>();
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            InplaceRsqrt(xLocal, TILE_LENGTH);
            AscendC::DataCopy(xGm[i * TILE_LENGTH], xLocal, TILE_LENGTH);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf;
    AscendC::GlobalTensor<half> xGm;
};

extern "C" __global__ __aicore__ void inplace_rsqrt_custom(GM_ADDR x)
{
    KernelInplaceRsqrt op;
    op.Init(x);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void inplace_rsqrt_custom_do(uint32_t blockDim, void *stream, uint8_t *x)
{
    inplace_rsqrt_custom<<<blockDim, nullptr, stream>>>(x);
}
#endif
