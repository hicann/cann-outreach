#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}
}

template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t baseBlockLength = totalLength / blockNum;
        const uint32_t tailBlockCount = totalLength % blockNum;
        const uint32_t extra = blockIdx < tailBlockCount ? 1U : 0U;

        blockLength_ = baseBlockLength + extra;
        tileNum_ = blockLength_ / TILE_LENGTH;
        tailLength_ = blockLength_ % TILE_LENGTH;

        const uint32_t blockOffset =
            blockIdx * baseBlockLength + (blockIdx < tailBlockCount ? blockIdx : tailBlockCount);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z) + blockOffset, blockLength_);

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(zQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        for (uint32_t tileIdx = 0; tileIdx < tileNum_; ++tileIdx) {
            CopyIn(tileIdx);
            Compute();
            CopyOut(tileIdx);
        }
        if (tailLength_ != 0) {
            ProcessTail();
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        AscendC::LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        AscendC::DataCopy(xLocal, xGm_[offset], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm_[offset], TILE_LENGTH);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = zQueue_.AllocTensor<T>();
        AscendC::Sub(zLocal, xLocal, yLocal, TILE_LENGTH);
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        AscendC::LocalTensor<T> zLocal = zQueue_.DeQue<T>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        AscendC::DataCopy(zGm_[offset], zLocal, TILE_LENGTH);
        zQueue_.FreeTensor(zLocal);
    }

    __aicore__ inline void ProcessTail()
    {
        const uint32_t offset = tileNum_ * TILE_LENGTH;
        for (uint32_t i = 0; i < tailLength_; ++i) {
            const T xValue = xGm_.GetValue(offset + i);
            const T yValue = yGm_.GetValue(offset + i);
            const float result = static_cast<float>(xValue) - static_cast<float>(yValue);
            zGm_.SetValue(offset + i, static_cast<T>(result));
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQueue_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tailLength_ = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSubCustomTemplate<DTYPE_X> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}