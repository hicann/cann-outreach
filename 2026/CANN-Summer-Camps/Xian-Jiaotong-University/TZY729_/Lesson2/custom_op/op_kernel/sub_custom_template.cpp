#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / TILE_NUM / BUFFER_NUM;

        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z *)z + blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<TYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();

        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    KernelSubCustomTemplate<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}

