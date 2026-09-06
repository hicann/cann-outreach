// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"
constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        tileLength = blockLength / tileNum / BUFFER_NUM;
        const uint32_t offset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, blockLength);
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum * BUFFER_NUM; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = xQueue.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = xQueue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = zQueue.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, tileLength);
        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = zQueue.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        zQueue.FreeTensor(zLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue, yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQueue;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
