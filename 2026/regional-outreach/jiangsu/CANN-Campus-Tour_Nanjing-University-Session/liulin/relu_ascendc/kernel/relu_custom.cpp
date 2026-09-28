/*
 * Relu 算子 kernel 侧实现（AscendC）
 * 输入 : x  float16, ND 格式, 4 维 [N4, N3, N2, N1]
 * 输出 : y  float16, 同 shape
 * 逻辑 : 按一维总元素量 total = N4*N3*N2*N1 分块，逐 tile 执行 Max(x, 0)
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalNum)
    {
        this->totalNum = totalNum;

        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * totalNum, totalNum);
        yGm.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * totalNum, totalNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_SIZE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = totalNum / TILE_SIZE;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        // relu: y = max(x, 0)
        AscendC::Max(yLocal, xLocal, static_cast<half>(0.0f), TILE_SIZE);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * TILE_SIZE], yLocal, TILE_SIZE);
        outQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr int32_t TILE_SIZE = 2048; // 每 tile 元素数, 需为 16 的倍数
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm, yGm;
    uint32_t totalNum = 0;
};

// 核函数入口: 4 维 shape 在 host 侧已折算为 totalNum
extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelRelu op;
    op.Init(x, y, tilingData.totalNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
