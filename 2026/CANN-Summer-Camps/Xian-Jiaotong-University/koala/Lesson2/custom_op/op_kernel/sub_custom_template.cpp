#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t blockIdx) {
        // Lesson2固定8核，用GetBlockNum()自动适配，不硬编码
        uint32_t blockNum = GetBlockNum();
        uint32_t avgLength = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        uint32_t blockLength = avgLength;
        uint32_t offset = blockIdx * avgLength;
        
        // 最后一个核处理余数，不丢元素
        if (blockIdx == blockNum - 1) {
            blockLength = avgLength + remainder;
        }
        
        xGm.SetGlobalBuffer((__gm__ half*)(x + offset * sizeof(half)), blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)(y + offset * sizeof(half)), blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)(z + offset * sizeof(half)), blockLength);
        
        pipe.InitBuffer(xQueue, 1, blockLength * sizeof(half));
        pipe.InitBuffer(yQueue, 1, blockLength * sizeof(half));
        pipe.InitBuffer(zQueue, 1, blockLength * sizeof(half));
        
        this->blockLength = blockLength;
    }
    
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<half> xLocal = xQueue.AllocTensor<half>();
        LocalTensor<half> yLocal = yQueue.AllocTensor<half>();
        DataCopy(xLocal, xGm, blockLength);
        DataCopy(yLocal, yGm, blockLength);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute() {
        LocalTensor<half> xLocal = xQueue.DeQue<half>();
        LocalTensor<half> yLocal = yQueue.DeQue<half>();
        LocalTensor<half> zLocal = zQueue.AllocTensor<half>();
        Sub(zLocal, xLocal, yLocal, blockLength);
        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut() {
        LocalTensor<half> zLocal = zQueue.DeQue<half>();
        DataCopy(zGm, zLocal, blockLength);
        zQueue.FreeTensor(zLocal);
    }

private:
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> zQueue;
    TPipe pipe;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    uint32_t blockIdx = GetBlockIdx();
    // Lesson2的Tiling字段是size
    KernelSub op;
    op.Init(x, y, z, tilingData.size, blockIdx);
    op.Process();
}