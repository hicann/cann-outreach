#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhCustomTilingData* tilingData)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x);
        yGm.SetGlobalBuffer((__gm__ half *)y);
        this->tileNum = tilingData->tileNum;
        uint32_t totalLength = tilingData->totalLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());
        this->blockLength = totalLength / blockDim;

        this->tileLength = this->blockLength / this->tileNum;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        uint32_t offsetInBlock = (progress % tileNum) * tileLength;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t gmOffset = blockIdx * blockLength + offsetInBlock;
        AscendC::DataCopy(xLocal, xGm[gmOffset], tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        AscendC::Tanh(yLocal, xLocal, tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        uint32_t offsetInBlock = (progress % tileNum) * tileLength;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t gmOffset = blockIdx * blockLength + offsetInBlock;
        AscendC::DataCopy(yGm[gmOffset], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh op;
    op.Init(x, y, &tilingData);
    op.Process();
}
