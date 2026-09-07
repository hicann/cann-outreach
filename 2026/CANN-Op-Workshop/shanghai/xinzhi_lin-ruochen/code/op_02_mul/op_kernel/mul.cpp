// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const MulTilingData &tiling) {
        tileLength_ = tiling.tileLength;
        const uint32_t blockOffset = AscendC::GetBlockIdx() * tiling.blockLength;
        if (blockOffset >= tiling.length) {
            return;
        }
        const uint32_t remaining = tiling.length - blockOffset;
        blockLength_ = remaining < tiling.blockLength ? remaining : tiling.blockLength;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, blockLength_);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        uint32_t offset = 0;
        while (offset < blockLength_) {
            const uint32_t remaining = blockLength_ - offset;
            const uint32_t count = remaining < tileLength_ ? remaining : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.template AllocTensor<DT_X>();
        if (count % ALIGN_NUM == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
            AscendC::DataCopy(yLocal, yGm_[offset], count);
        } else {
            // Read exactly the valid bytes, then pad only inside UB.
            const AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(ALIGN_NUM - count % ALIGN_NUM),
                static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);
        }
        inQueueX_.template EnQue<DT_X>(xLocal);
        inQueueY_.template EnQue<DT_X>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.template AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, static_cast<int32_t>(count));

        outQueueZ_.template EnQue<DT_X>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.template DeQue<DT_X>();
        if (count % ALIGN_NUM == 0) {
            AscendC::DataCopy(zGm_[offset], zLocal, count);
        } else {
            const AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        }
        outQueueZ_.FreeTensor(zLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    if (tiling_data.length == 0) {
        return;
    }
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
