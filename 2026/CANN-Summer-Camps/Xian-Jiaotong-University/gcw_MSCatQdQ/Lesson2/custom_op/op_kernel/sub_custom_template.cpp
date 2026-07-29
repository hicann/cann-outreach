#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        blockLength = totalLength / blockNum;
        tileLength = blockLength / TILE_NUM / BUFFER_NUM;

        const uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        auto xLocal = inQueueX.AllocTensor<dtypeX>();
        auto yLocal = inQueueY.AllocTensor<dtypeY>();
        const uint32_t offset = progress * tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        auto xLocal = inQueueX.DeQue<dtypeX>();
        auto yLocal = inQueueY.DeQue<dtypeY>();
        auto zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        auto zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength{0};
    uint32_t tileLength{0};
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}