// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData *tilingData, AscendC::TPipe *pipe)
    {
        const uint64_t blockOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tilingData->blockLength;
        if (blockOffset >= tilingData->length || tilingData->tileLength == 0) {
            return;
        }

        const uint64_t remaining = tilingData->length - blockOffset;
        blockLength_ = static_cast<uint32_t>(
            (remaining < tilingData->blockLength) ? remaining : tilingData->blockLength);
        tileLength_ = tilingData->tileLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + blockOffset, blockLength_);

        pipe->InitBuffer(xQueue_, 2, tileLength_ * sizeof(DT_X));
        pipe->InitBuffer(yQueue_, 2, tileLength_ * sizeof(DT_X));
        pipe->InitBuffer(zQueue_, 2, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        uint32_t progress = 0;
        while (progress < blockLength_) {
            const uint32_t remaining = blockLength_ - progress;
            const uint32_t currentLength = (remaining < tileLength_) ? remaining : tileLength_;
            CopyIn(progress, currentLength);
            Compute(currentLength);
            CopyOut(progress, currentLength);
            progress += currentLength;
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.template AllocTensor<DT_X>();
        const AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(xLocal, xGm_[progress], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm_[progress], copyParams, padParams);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = zQueue_.template AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, static_cast<int32_t>(currentLength));
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_X> zLocal = zQueue_.template DeQue<DT_X>();
        const AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[progress], zLocal, copyParams);
        zQueue_.FreeTensor(zLocal);
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelMul<DT_X> op;
    op.Init(x, y, z, &tiling_data, &pipe);
    op.Process();
}
