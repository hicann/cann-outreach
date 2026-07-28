#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        blockLength_ = totalLength / AscendC::GetBlockNum();
        tileNum_ = tileNum;
        tileLength_ = blockLength_ / tileNum_;

        const uint32_t blockOffset = blockLength_ * AscendC::GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ T*)z + blockOffset, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm_[progress * tileLength_], tileLength_);
        AscendC::DataCopy(yLocal, yGm_[progress * tileLength_], tileLength_);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength_);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<T> zLocal = outQueueZ_.DeQue<T>();
        AscendC::DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> zGm_;
    uint32_t blockLength_;
    uint32_t tileNum_;
    uint32_t tileLength_;
};

template <typename T>
__aicore__ inline void RunSub(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              const SubCustomTemplateTilingData& tilingData)
{
    KernelSubCustomTemplate<T> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        RunSub<float>(x, y, z, tilingData);
    } else {
        RunSub<half>(x, y, z, tilingData);
    }
}
