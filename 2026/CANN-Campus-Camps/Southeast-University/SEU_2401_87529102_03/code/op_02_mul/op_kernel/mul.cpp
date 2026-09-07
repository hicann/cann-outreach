// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t length, uint32_t tileLength) {
        tileLength_ = tileLength;
        constexpr uint32_t alignElements = 32 / sizeof(DT_X);
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(length) + alignElements - 1) / alignElements);
        const uint32_t coreCount = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t base = blocks / coreCount;
        const uint32_t extra = blocks % coreCount;
        const uint32_t startBlock = core * base + (core < extra ? core : extra);
        const uint64_t offset = static_cast<uint64_t>(startBlock) * alignElements;
        const uint64_t capacity = static_cast<uint64_t>(base + (core < extra ? 1 : 0)) * alignElements;
        const uint64_t remaining = offset < length ? length - offset : 0;
        blockLength_ = static_cast<uint32_t>(remaining < capacity ? remaining : capacity);
        if (blockLength_ == 0) {
            return;
        }
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + offset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + offset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + offset, blockLength_);
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
        auto xLocal = inQueueX_.AllocTensor<DT_X>();
        auto yLocal = inQueueY_.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[offset], params, padding);
        AscendC::DataCopyPad(yLocal, yGm_[offset], params, padding);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t count) {
        auto xLocal = inQueueX_.DeQue<DT_X>();
        auto yLocal = inQueueY_.DeQue<DT_X>();
        auto zLocal = outQueueZ_.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, count);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        auto zLocal = outQueueZ_.DeQue<DT_X>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, params);
        outQueueZ_.FreeTensor(zLocal);
    }
    static constexpr int32_t BUFFER_NUM = 2;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileLength);
    op.Process();
}
