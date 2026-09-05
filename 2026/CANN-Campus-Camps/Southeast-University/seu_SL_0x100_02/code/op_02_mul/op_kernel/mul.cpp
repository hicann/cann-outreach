// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2; // 双缓冲

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length,
                                uint32_t blockLength, uint32_t tileLength) {
        blockLength_ = blockLength;
        tileLength_ = tileLength;
        // 本核处理的数据起点和实际长度(最后一核可能不足blockLength)
        uint32_t start = AscendC::GetBlockIdx() * blockLength_;
        actualLength_ = (length - start < blockLength_) ? (length - start) : blockLength_;
        // 本核实际循环次数
        tileNum_ = (actualLength_ + tileLength_ - 1) / tileLength_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start, actualLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start, actualLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + start, actualLength_);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    // 第index次循环实际处理的元素个数(最后一次可能不足tileLength)
    __aicore__ inline uint32_t GetTileLength(uint32_t index) {
        uint32_t remain = actualLength_ - index * tileLength_;
        return remain < tileLength_ ? remain : tileLength_;
    }
    __aicore__ inline void CopyIn(uint32_t index) {
        uint32_t copyLength = GetTileLength(index);
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm_[index * tileLength_], copyLength);
        AscendC::DataCopy(yLocal, yGm_[index * tileLength_], copyLength);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t index) {
        uint32_t copyLength = GetTileLength(index);
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, copyLength);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t index) {
        uint32_t copyLength = GetTileLength(index);
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.DeQue<DT_X>();
        AscendC::DataCopy(zGm_[index * tileLength_], zLocal, copyLength);
        outQueueZ_.FreeTensor(zLocal);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;
    uint32_t blockLength_;
    uint32_t tileLength_;
    uint32_t tileNum_;
    uint32_t actualLength_;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
