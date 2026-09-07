#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               const MulTilingData& tiling)
    {
        tileLength_ = tiling.tileLength;
        const uint32_t core = AscendC::GetBlockIdx();
        // 以完整 tile 为单位均分，前 remainder 个核各多处理一个 tile。
        const uint32_t tiles = tiling.tileNum;
        const uint32_t base = tiles / tiling.coreNum;
        const uint32_t remainder = tiles % tiling.coreNum;
        const uint32_t firstTile = core * base +
            (core < remainder ? core : remainder);
        const uint32_t ownedTiles = base + (core < remainder ? 1 : 0);
        const uint64_t offset = static_cast<uint64_t>(firstTile) * tileLength_;
        const uint64_t capacity = static_cast<uint64_t>(ownedTiles) * tileLength_;
        const uint64_t remaining = offset < tiling.totalLength ?
            tiling.totalLength - offset : 0;
        blockLength_ = static_cast<uint32_t>(
            capacity < remaining ? capacity : remaining);
        if (blockLength_ == 0) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z) + offset, blockLength_);
        pipe_.InitBuffer(inX_, 2, tileLength_ * sizeof(T));
        pipe_.InitBuffer(inY_, 2, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outZ_, 2, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_;) {
            const uint32_t remaining = blockLength_ - offset;
            const uint32_t count = remaining < tileLength_ ? remaining : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        auto xLocal = inX_.AllocTensor<T>();
        auto yLocal = inY_.AllocTensor<T>();
        if ((count * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
            AscendC::DataCopy(yLocal, yGm_[offset], count);
        } else {
            const uint32_t align = 32 / sizeof(T);
            AscendC::DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0, static_cast<uint8_t>(align - count % align), 0};
            AscendC::DataCopyPad(xLocal, xGm_[offset], copy, pad);
            AscendC::DataCopyPad(yLocal, yGm_[offset], copy, pad);
        }
        inX_.EnQue(xLocal);
        inY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = inX_.DeQue<T>();
        auto yLocal = inY_.DeQue<T>();
        auto zLocal = outZ_.AllocTensor<T>();
        AscendC::Mul(zLocal, xLocal, yLocal, count);
        outZ_.EnQue(zLocal);
        inX_.FreeTensor(xLocal);
        inY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        auto zLocal = outZ_.DeQue<T>();
        if ((count * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(zGm_[offset], zLocal, count);
        } else {
            // 只写有效字节，避免尾块越界或覆盖其他核的输出。
            AscendC::DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[offset], zLocal, copy);
        }
        outZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inX_, inY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outZ_;
    AscendC::GlobalTensor<T> xGm_, yGm_, zGm_;
    uint32_t tileLength_ = 0;
    uint32_t blockLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData);
    op.Process();
}
