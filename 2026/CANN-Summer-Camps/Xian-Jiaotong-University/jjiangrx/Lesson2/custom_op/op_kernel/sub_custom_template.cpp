#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename DtypeX, typename DtypeY, typename DtypeZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        tileLength = blockLength / tileNum / BUFFER_NUM;

        const uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DtypeX *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DtypeY *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DtypeZ *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DtypeZ));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        const uint32_t offset = progress * tileLength;
        AscendC::LocalTensor<DtypeX> xLocal = inQueueX.AllocTensor<DtypeX>();
        AscendC::LocalTensor<DtypeY> yLocal = inQueueY.AllocTensor<DtypeY>();
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DtypeX> xLocal = inQueueX.DeQue<DtypeX>();
        AscendC::LocalTensor<DtypeY> yLocal = inQueueY.DeQue<DtypeY>();
        AscendC::LocalTensor<DtypeZ> zLocal = outQueueZ.AllocTensor<DtypeZ>();
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        const uint32_t offset = progress * tileLength;
        AscendC::LocalTensor<DtypeZ> zLocal = outQueueZ.DeQue<DtypeZ>();
        AscendC::DataCopy(zGm[offset], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DtypeX> xGm;
    AscendC::GlobalTensor<DtypeY> yGm;
    AscendC::GlobalTensor<DtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tilingData, tiling);
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
