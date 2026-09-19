#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t tileLength)
    {
        tileLength_ = tileLength;
        const uint32_t blockLength = length / AscendC::GetBlockNum();
        const uint32_t offset = AscendC::GetBlockIdx() * blockLength;
        tileCount_ = blockLength / tileLength_;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + offset, blockLength);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + offset, blockLength);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + offset, blockLength);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileCount_; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY_.AllocTensor<DT_X>();

        const uint32_t offset = progress * tileLength_;
        AscendC::DataCopy(xLocal, xGm_[offset], tileLength_);
        AscendC::DataCopy(yLocal, yGm_[offset], tileLength_);

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ_.AllocTensor<DT_X>();

        AscendC::Add(zLocal, xLocal, yLocal, tileLength_);

        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ_.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm_[progress * tileLength_], zLocal, tileLength_);

        outQueueZ_.FreeTensor(zLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;

    uint32_t tileLength_;
    uint32_t tileCount_;
};

template <typename DT_X>
__global__ __aicore__ void add(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);

    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileLength);
    op.Process();
}