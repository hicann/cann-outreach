#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;   
constexpr uint32_t TILE_NUM   = 8;  

class SubCustomTemplate {
public:
    __aicore__ inline SubCustomTemplate() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t size)
    {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        this->blockLength = size / blockNum;              
        this->tileLength  = this->blockLength / TILE_NUM / BUFFER_NUM; 

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + this->blockLength * blockIdx, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + this->blockLength * blockIdx, this->blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(z) + this->blockLength * blockIdx, this->blockLength);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;  // 16
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
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Sub(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN,  BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}