#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using DTYPE = half;
constexpr int32_t BUFFER_NUM = 2;

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        this->blockLength = totalLength / blockNum;
        this->tileLength = 1024;
        this->loopCount = (blockLength + tileLength - 1) / tileLength;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DTYPE*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE*)z + offset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DTYPE));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < loopCount; i++) {
            int32_t currentLen = (i == loopCount - 1) ? (blockLength - i * tileLength) : tileLength;
            CopyIn(i, currentLen);
            Compute(currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t currentLen)
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = inQueueY.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], currentLen);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], currentLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t currentLen)
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = inQueueY.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        AscendC::Sub(zLocal, xLocal, yLocal, currentLen);
        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, int32_t currentLen)
    {
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.DeQue<DTYPE>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, currentLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileLength;
    int32_t loopCount;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSub op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}