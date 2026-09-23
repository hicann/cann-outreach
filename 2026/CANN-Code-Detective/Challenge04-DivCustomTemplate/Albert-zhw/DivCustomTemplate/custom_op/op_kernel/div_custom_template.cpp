#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;
}

template <typename TX, typename TY, typename TZ>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        this->blockLength = totalLength / blockNum;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ TX*)x + this->blockLength * blockIdx, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ TY*)y + this->blockLength * blockIdx, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TZ*)z + this->blockLength * blockIdx, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(TY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(TZ));
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
        AscendC::LocalTensor<TX> xLocal = inQueueX.AllocTensor<TX>();
        AscendC::LocalTensor<TY> yLocal = inQueueY.AllocTensor<TY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TX> xLocal = inQueueX.DeQue<TX>();
        AscendC::LocalTensor<TY> yLocal = inQueueY.DeQue<TY>();
        AscendC::LocalTensor<TZ> zLocal = outQueueZ.AllocTensor<TZ>();
        AscendC::Div(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<TZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TZ> zLocal = outQueueZ.DeQue<TZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TX> xGm;
    AscendC::GlobalTensor<TY> yGm;
    AscendC::GlobalTensor<TZ> zGm;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelDiv<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size, 8);
    op.Process();
}
