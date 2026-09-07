// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE_LENGTH = 256;
constexpr uint32_t BUFFER_NUM = 2;
}

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());

        // TilingFunc保证length可以被blockNum整除，因此每个核处理相同数量的数据。
        blockLength_ = (blockNum == 0) ? 0 : length / blockNum;
        offset_ = blockIdx * blockLength_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + offset_, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + offset_, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + offset_, blockLength_);

        // 双缓冲：每个队列两个tile。
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t offset = 0; offset < blockLength_; offset += TILE_LENGTH) {
            uint32_t tileLength = blockLength_ - offset;
            if (tileLength > TILE_LENGTH) {
                tileLength = TILE_LENGTH;
            }

            CopyIn(offset, tileLength);
            Compute(tileLength);
            CopyOut(offset, tileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t tileLength) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY_.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm_[offset], tileLength);
        DataCopy(yLocal, yGm_[offset], tileLength);

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t tileLength) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY_.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ_.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, tileLength);

        outQueueZ_.EnQue<DT_X>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t tileLength) {
        LocalTensor<DT_X> zLocal = outQueueZ_.DeQue<DT_X>();
        DataCopy(zGm_[offset], zLocal, tileLength);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    GlobalTensor<DT_X> zGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;

    uint32_t offset_ = 0;
    uint32_t blockLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
