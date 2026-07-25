#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#ifndef DTYPE_Z
#define DTYPE_Z half
#endif

constexpr int32_t BUFFER_NUM = 2;
namespace {
constexpr uint32_t TILE_NUM_PER_CORE = 8;
}

template <typename XType, typename YType, typename ZType>
class SubKernel {
public:
    __aicore__ inline SubKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength)
    {
        const uint32_t blockCount = AscendC::GetBlockNum();
        const uint32_t blockIndex = AscendC::GetBlockIdx();

        blockLength_ = totalLength / blockCount;
        tileLength_ = blockLength_ / TILE_NUM_PER_CORE;

        const uint32_t blockOffset = blockIndex * blockLength_;

        xGm_.SetGlobalBuffer(
            (__gm__ XType*)x + blockOffset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            (__gm__ YType*)y + blockOffset,
            blockLength_);

        zGm_.SetGlobalBuffer(
            (__gm__ ZType*)z + blockOffset,
            blockLength_);

        pipe_.InitBuffer(
            xQueue_,
            BUFFER_NUM,
            tileLength_ * sizeof(XType));

        pipe_.InitBuffer(
            yQueue_,
            BUFFER_NUM,
            tileLength_ * sizeof(YType));

        pipe_.InitBuffer(
            zQueue_,
            BUFFER_NUM,
            tileLength_ * sizeof(ZType));
    }

    __aicore__ inline void Run()
    {
        uint32_t tileIndex = 0;

        while (tileIndex < TILE_NUM_PER_CORE) {
            CopyIn(tileIndex);
            Subtract();
            CopyOut(tileIndex);
            ++tileIndex;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIndex)
    {
        const uint32_t offset = tileIndex * tileLength_;

        AscendC::LocalTensor<XType> xLocal =
            xQueue_.AllocTensor<XType>();

        AscendC::LocalTensor<YType> yLocal =
            yQueue_.AllocTensor<YType>();

        AscendC::DataCopy(
            xLocal,
            xGm_[offset],
            tileLength_);

        AscendC::DataCopy(
            yLocal,
            yGm_[offset],
            tileLength_);

        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Subtract()
    {
        AscendC::LocalTensor<XType> xLocal =
            xQueue_.DeQue<XType>();

        AscendC::LocalTensor<YType> yLocal =
            yQueue_.DeQue<YType>();

        AscendC::LocalTensor<ZType> zLocal =
            zQueue_.AllocTensor<ZType>();

        AscendC::Sub(
            zLocal,
            xLocal,
            yLocal,
            tileLength_);

        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
        zQueue_.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIndex)
    {
        const uint32_t offset = tileIndex * tileLength_;

        AscendC::LocalTensor<ZType> zLocal =
            zQueue_.DeQue<ZType>();

        AscendC::DataCopy(
            zGm_[offset],
            zLocal,
            tileLength_);

        zQueue_.FreeTensor(zLocal);
    }

private:
    uint32_t blockLength_;
    uint32_t tileLength_;

    AscendC::TPipe pipe_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> xQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> yQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> zQueue_;

    AscendC::GlobalTensor<XType> xGm_;
    AscendC::GlobalTensor<YType> yGm_;
    AscendC::GlobalTensor<ZType> zGm_;
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

    SubKernel<DTYPE_X, DTYPE_Y, DTYPE_Z> kernel;
    kernel.Init(x, y, z, tilingData.size);
    kernel.Run();
}