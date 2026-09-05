#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length, uint32_t tileLength) {
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t baseLength = length / blockNum;
        uint32_t remainder = length % blockNum;

        blockLength = baseLength + (blockIdx < remainder ? 1 : 0);
        uint32_t blockOffset = blockIdx * baseLength + (blockIdx < remainder ? blockIdx : remainder);
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockLength == 0) return;
        uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;

        for (uint32_t i = 0; i < tileNum; ++i) {
            uint32_t offset = i * tileLength;
            uint32_t currentLength = blockLength - offset;
            if (currentLength > tileLength) currentLength = tileLength;
            CopyIn(i, currentLength);
            Compute(currentLength);
            CopyOut(i, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        uint32_t offset = progress * tileLength;

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentLength * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};

        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, currentLength);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentLength) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        uint32_t offset = progress * tileLength;

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentLength * sizeof(DT_X)),
            0,
            0,
            0
        };

        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.length, tilingData.tileLength);
    op.Process();
}