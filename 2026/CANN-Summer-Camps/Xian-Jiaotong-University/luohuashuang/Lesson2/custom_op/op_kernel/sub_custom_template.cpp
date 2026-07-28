#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_SIZE = 2048;

template<typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        ASSERT(blockNum != 0 && "block dim can not be zero!");

        const uint32_t baseLength = totalLength / blockNum;
        const uint32_t remainder = totalLength % blockNum;
        this->blockLength = baseLength + (blockIdx < remainder ? 1U : 0U);
        this->blockOffset = blockIdx * baseLength + (blockIdx < remainder ? blockIdx : remainder);

        if (this->blockLength == 0) {
            return;
        }

        this->tileNum = (this->blockLength + TILE_SIZE - 1) / TILE_SIZE;

        xGm.SetGlobalBuffer((__gm__ T*)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t copyLength = (i == this->tileNum - 1) 
                ? (this->blockLength - i * TILE_SIZE) 
                : TILE_SIZE;
            uint32_t offset = i * TILE_SIZE;
            CopyIn(offset, copyLength);
            Compute(copyLength);
            CopyOut(offset, copyLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t copyLength)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], copyLength);
        DataCopy(yLocal, yGm[offset], copyLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t copyLength)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        Sub(zLocal, xLocal, yLocal, copyLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t copyLength)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[offset], zLocal, copyLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength = 0;
    uint32_t blockOffset = 0;
    uint32_t tileNum = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template_fp16(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    KernelSubCustomTemplate<half> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}

extern "C" __global__ __aicore__ void sub_custom_template_fp32(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    KernelSubCustomTemplate<float> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}