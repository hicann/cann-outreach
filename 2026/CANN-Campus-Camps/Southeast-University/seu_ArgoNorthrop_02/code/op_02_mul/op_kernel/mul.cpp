// Kernel侧核函数实现 —— z = x * y
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM  = 2;
constexpr uint32_t TILE_LENGTH = 128;  // 每块固定元素数（128*4=512B，32B对齐）

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // Pipe & Queue
        pipe.InitBuffer(queueX_, BUFFER_NUM, sizeof(DT_X) * TILE_LENGTH);
        pipe.InitBuffer(queueY_, BUFFER_NUM, sizeof(DT_X) * TILE_LENGTH);
        pipe.InitBuffer(queueZ_, BUFFER_NUM, sizeof(DT_X) * TILE_LENGTH);

        // GlobalTensor
        xGm_.SetGlobalBuffer((__gm__ DT_X*)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y);
        zGm_.SetGlobalBuffer((__gm__ DT_X*)z);

        // Tiling —— 32B 对齐 perCore，杜绝最后一块尾巴越界
        uint32_t alignElements = (uint32_t)(32 / sizeof(DT_X));  // fp32=8, fp16=16

        blockNum_ = AscendC::GetBlockNum();
        blockIdx_ = (uint32_t)AscendC::GetBlockIdx();

        // 先算每核平均，再向上对齐到 alignElements（局部变量，init 内部用）
        uint32_t perCoreRaw = (length + blockNum_ - 1) / blockNum_;
        uint32_t perCore = ((perCoreRaw + alignElements - 1) / alignElements) * alignElements;

        coreOffset_ = perCore * blockIdx_;
        if (coreOffset_ >= length) {
            coreLen_ = 0;
        } else {
            uint32_t coreLenRaw = (coreOffset_ + perCore > length)
                                   ? (length - coreOffset_)
                                   : perCore;
            coreLen_ = ((coreLenRaw + alignElements - 1) / alignElements) * alignElements;
        }

        // 这个核需要分几块处理
        tileNum_ = (coreLen_ + TILE_LENGTH - 1) / TILE_LENGTH;
    }

    __aicore__ inline void Process() {
        if (coreLen_ == 0) {
            return;
        }
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute();
            CopyOut();
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> queueX_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> queueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> queueZ_;

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;

    uint32_t blockNum_   = 0;
    uint32_t blockIdx_   = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreLen_    = 0;
    uint32_t tileNum_    = 0;

    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = coreOffset_ + tileIdx * TILE_LENGTH;
        // 最后一块可能不足 TILE_LENGTH
        uint32_t realLen = (offset + TILE_LENGTH > coreOffset_ + coreLen_)
                           ? (coreOffset_ + coreLen_ - offset)
                           : TILE_LENGTH;
        currentRealLen_ = realLen;

        AscendC::LocalTensor<DT_X> xLocal = queueX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = queueY_.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm_[offset], realLen);
        AscendC::DataCopy(yLocal, yGm_[offset], realLen);

        queueX_.EnQue(xLocal);
        queueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = queueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = queueY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = queueZ_.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, currentRealLen_);

        queueZ_.EnQue(zLocal);
        queueX_.FreeTensor(xLocal);
        queueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<DT_X> zLocal = queueZ_.DeQue<DT_X>();

        uint32_t offset = coreOffset_ + currentOutIdx_ * TILE_LENGTH;
        AscendC::DataCopy(zGm_[offset], zLocal, currentRealLen_);

        queueZ_.FreeTensor(zLocal);
        currentOutIdx_++;
    }

    uint32_t currentOutIdx_  = 0;
    uint32_t currentRealLen_ = 0;
};

// ========== 关键：恢复官方 tiling key 模式 ==========
// 模板实例分发由 tiling_key_mul.h 里的 ASCENDC_TPL_SEL 自动处理
template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
