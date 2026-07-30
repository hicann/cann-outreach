#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_SIZE = 16;

using namespace AscendC;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd
{
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockIdx  = GetBlockIdx();
        uint32_t blockNum  = GetBlockNum();
        blockLength        = totalLength / blockNum;
        blockOffset        = blockLength * blockIdx;
        this->tileNum      = tileNum;

        // 采用第二份对齐tileLength计算逻辑
        tileLength = (blockLength / tileNum / ALIGN_SIZE) * ALIGN_SIZE;
        if (tileLength < ALIGN_SIZE)
        {
            tileLength = (blockLength < ALIGN_SIZE) ? blockLength : ALIGN_SIZE;
        }

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        // 使用第二份代码的流水重叠调度逻辑，CopyIn‑Compute‑CopyOut并行重叠
        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        CopyIn(0);
        for (int32_t i = 0; i < loopCount - 1; ++i)
        {
            CopyIn(i + 1);
            Compute(i);
            CopyOut(i);
        }
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();

        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        DataCopy(yLocal, yGm[progress * tileLength], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();

        Add(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<dtypeX> xGm;
    GlobalTensor<dtypeY> yGm;
    GlobalTensor<dtypeZ> zGm;

    uint32_t blockOffset;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__
void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);

    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}