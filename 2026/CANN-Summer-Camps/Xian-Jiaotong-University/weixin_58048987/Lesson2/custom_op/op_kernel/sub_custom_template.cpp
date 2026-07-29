#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

template <typename Tx, typename Ty, typename Tz>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        blockLength = totalLength / AscendC::GetBlockNum();
        tileLength = blockLength / TILE_NUM / BUFFER_NUM;
        const uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ Tx *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ Ty *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ Tz *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(Tx));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(Ty));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(Tz));
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<Tx> xLocal = inQueueX.AllocTensor<Tx>();
        AscendC::LocalTensor<Ty> yLocal = inQueueY.AllocTensor<Ty>();
        const uint32_t offset = static_cast<uint32_t>(progress) * tileLength;

        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<Tx> xLocal = inQueueX.DeQue<Tx>();
        AscendC::LocalTensor<Ty> yLocal = inQueueY.DeQue<Ty>();
        AscendC::LocalTensor<Tz> zLocal = outQueueZ.AllocTensor<Tz>();

        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue<Tz>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<Tz> zLocal = outQueueZ.DeQue<Tz>();
        const uint32_t offset = static_cast<uint32_t>(progress) * tileLength;
        AscendC::DataCopy(zGm[offset], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<Tx> xGm;
    AscendC::GlobalTensor<Ty> yGm;
    AscendC::GlobalTensor<Tz> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
