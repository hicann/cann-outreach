// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength) {
        this->tileLength = tileLength;
        uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();
        uint32_t remainingLength = blockOffset < totalLength ? totalLength - blockOffset : 0;
        this->blockLength = blockOffset < totalLength
            ? (blockLength < remainingLength ? blockLength : remainingLength)
            : 0;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        uint32_t loopCount = (blockLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * tileLength;
            uint32_t remainingLength = blockLength - offset;
            uint32_t currentLength = tileLength < remainingLength ? tileLength : remainingLength;
            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        uint32_t copyBytes = static_cast<uint32_t>(length * sizeof(DT_X));
        AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        uint32_t copyBytes = static_cast<uint32_t>(length * sizeof(DT_X));
        AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}