#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
    using namespace AscendC;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    
    uint32_t totalLength = tilingData.size;
    ASSERT(GetBlockNum() != 0);
    
    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();
    
    // 计算均分长度和余数
    uint32_t baseBlockLength = totalLength / blockNum;
    uint32_t tailLength = totalLength % blockNum;
    
    uint32_t blockLength = baseBlockLength;
    uint32_t globalOffset = baseBlockLength * blockIdx;
    
    // 将余数分配给最后一个 block
    if (blockIdx == blockNum - 1) {
        blockLength += tailLength;
    }
    
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;
    
    // 使用动态计算出的 globalOffset 和 blockLength
    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalOffset, blockLength);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalOffset, blockLength);
    zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + globalOffset, blockLength);
    
    pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(DTYPE_X));
    pipe.InitBuffer(inQueueY, 1, blockLength * sizeof(DTYPE_Y));
    pipe.InitBuffer(outQueueZ, 1, blockLength * sizeof(DTYPE_Z));
    
    LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
    
    DataCopy(xLocal, xGm, blockLength);
    DataCopy(yLocal, yGm, blockLength);
    
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
    
    xLocal = inQueueX.DeQue<DTYPE_X>();
    yLocal = inQueueY.DeQue<DTYPE_Y>();
    
    Sub(zLocal, xLocal, yLocal, blockLength);
    
    outQueueZ.EnQue(zLocal);
    zLocal = outQueueZ.DeQue<DTYPE_Z>();
    
    DataCopy(zGm, zLocal, blockLength);
    
    outQueueZ.FreeTensor(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}