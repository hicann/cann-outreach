/**
 * @file relu_custom.cpp
 * @brief Ascend C Relu operator: y = max(x, 0)
 *        Shape: 4D [N4, N3, N2, N1] = [2, 2, 4, 128]
 *        Dtype: float16 (half), Format: ND
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 2 * 2 * 4 * 128; // 2048
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM; // 256
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // 64

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::Relu(yLocal, xLocal, TILE_LENGTH);
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[progress * TILE_LENGTH], yLocal, TILE_LENGTH);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y)
{
    KernelRelu op;
    op.Init(x, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void relu_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    relu_custom<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
