#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {
        this->tileNum = tileNum;
        this->blockLength = totalLength / GetBlockNum(); 
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        // 核心计算逻辑：0.0作为标量与输入比较求最大值 (float16 / half)
        half zero(0.0);
        Max(yLocal, xLocal, zero, this->tileLength);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
