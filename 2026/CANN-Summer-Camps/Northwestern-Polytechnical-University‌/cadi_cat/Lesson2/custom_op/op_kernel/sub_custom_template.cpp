#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    constexpr uint32_t BLOCK_DIM = 8;
    using T = half;

    const uint32_t blockLength = tilingData.size / BLOCK_DIM;
    const uint32_t blockOffset = AscendC::GetBlockIdx() * blockLength;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;
    xGm.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength);
    yGm.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength);
    zGm.SetGlobalBuffer((__gm__ T*)z + blockOffset, blockLength);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    pipe.InitBuffer(xQueue, 1, blockLength * sizeof(T));
    pipe.InitBuffer(yQueue, 1, blockLength * sizeof(T));
    pipe.InitBuffer(zQueue, 1, blockLength * sizeof(T));

    AscendC::LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm, blockLength);
    AscendC::DataCopy(yLocal, yGm, blockLength);
    xQueue.EnQue(xLocal);
    yQueue.EnQue(yLocal);

    xLocal = xQueue.DeQue<T>();
    yLocal = yQueue.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
    AscendC::Sub(zLocal, xLocal, yLocal, blockLength);
    zQueue.EnQue<T>(zLocal);
    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);

    zLocal = zQueue.DeQue<T>();
    AscendC::DataCopy(zGm, zLocal, blockLength);
    zQueue.FreeTensor(zLocal);
}
