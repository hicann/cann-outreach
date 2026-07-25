#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 256;
}

template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        // Host 侧设置 BlockDim = 8，因此每个核处理 16384 / 8 = 2048 个元素。
        blockLength_ = totalLength / GetBlockNum();
        loopCount_ = blockLength_ / TILE_LENGTH;

        const uint32_t blockOffset = GetBlockIdx() * blockLength_;

        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(x) + blockOffset,
            blockLength_);
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(y) + blockOffset,
            blockLength_);
        zGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(z) + blockOffset,
            blockLength_);

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(zQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < loopCount_; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();

        const uint32_t offset = progress * TILE_LENGTH;
        DataCopy(xLocal, xGm_[offset], TILE_LENGTH);
        DataCopy(yLocal, yGm_[offset], TILE_LENGTH);

        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        LocalTensor<T> zLocal = zQueue_.AllocTensor<T>();

        Sub(zLocal, xLocal, yLocal, TILE_LENGTH);

        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<T> zLocal = zQueue_.DeQue<T>();
        const uint32_t offset = progress * TILE_LENGTH;

        DataCopy(zGm_[offset], zLocal, TILE_LENGTH);
        zQueue_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> zGm_;

    uint32_t blockLength_ = 0;
    uint32_t loopCount_ = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSubCustomTemplate<DTYPE_X> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}