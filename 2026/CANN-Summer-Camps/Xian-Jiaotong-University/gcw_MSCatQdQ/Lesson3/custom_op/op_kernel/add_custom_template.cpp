#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t VECTOR_ALIGN_SIZE = 16;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd
{
public:
    __aicore__ inline KernelAdd() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        if (totalLength == 0 || tileNum == 0)
        {
            blockLen = 0;
            tileCnt = 0;
            tileSingleLen = 0;
            return;
        }

        const uint32_t blockNum = AscendC::GetBlockNum();
        blockLen = totalLength / blockNum;
        tileCnt = tileNum;

        uint32_t baseTile = blockLen / tileCnt;
        tileSingleLen = (baseTile / VECTOR_ALIGN_SIZE) * VECTOR_ALIGN_SIZE;

        if (tileSingleLen < VECTOR_ALIGN_SIZE)
        {
            tileSingleLen = (blockLen < VECTOR_ALIGN_SIZE) ? blockLen : VECTOR_ALIGN_SIZE;
        }

        const uint32_t blockOffset = blockLen * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, blockLen);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, blockLen);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, blockLen);

        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, tileSingleLen * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, DOUBLE_BUFFER_NUM, tileSingleLen * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, DOUBLE_BUFFER_NUM, tileSingleLen * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        if (blockLen == 0 || tileCnt == 0 || tileSingleLen == 0)
        {
            return;
        }

        CopyIn(0);

        for (int32_t i = 0; i < tileCnt - 1; i++)
        {
            CopyIn(i + 1);
            Compute(i);
            CopyOut(i);
        }

        Compute(tileCnt - 1);
        CopyOut(tileCnt - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * tileSingleLen], tileSingleLen);
        AscendC::DataCopy(yLocal, yGm[progress * tileSingleLen], tileSingleLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, tileSingleLen);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * tileSingleLen], zLocal, tileSingleLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLen;
    uint32_t tileCnt;
    uint32_t tileSingleLen;
};

__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}