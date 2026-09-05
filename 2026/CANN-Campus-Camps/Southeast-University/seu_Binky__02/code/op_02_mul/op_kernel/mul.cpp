#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t idx = AscendC::GetBlockIdx();
        uint32_t baseLen = totalLength / coreNum;
        uint32_t remain = totalLength % coreNum;
        uint32_t start = idx * baseLen + (idx < remain ? idx : remain);
        uint32_t len = baseLen + (idx < remain ? 1 : 0);
        this->blockLength = len;
        this->tileNum = tileNum;
        // 计算 tileLength，向上取整以保证覆盖所有数据
        this->tileLength = (this->blockLength + tileNum * BUFFER_NUM - 1) / (tileNum * BUFFER_NUM);
        if (this->tileLength == 0) this->tileLength = 1;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + start, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + start, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t totalTiles = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < totalTiles; i++) {
            uint32_t curLen = this->tileLength;
            // 最后一次可能不足
            if (i == totalTiles - 1) {
                curLen = this->blockLength - i * this->tileLength;
            }
            CopyIn(i, curLen);
            Compute(i, curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t len) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], len);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t len) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t len) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
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