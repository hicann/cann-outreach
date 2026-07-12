#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
    (void)workspace;

    constexpr int32_t BUFFER_NUM = 2;
    constexpr uint32_t TILE_NUM = 8;
    uint32_t blockLength = tilingData.size / AscendC::GetBlockNum();
    uint32_t tileLength = blockLength / TILE_NUM / BUFFER_NUM;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockLength * AscendC::GetBlockIdx(), blockLength);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + blockLength * AscendC::GetBlockIdx(), blockLength);
    zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + blockLength * AscendC::GetBlockIdx(), blockLength);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DTYPE_Z));

    for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[i * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[i * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        xLocal = inQueueX.DeQue<DTYPE_X>();
        yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[i * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
}
