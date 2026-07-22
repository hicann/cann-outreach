#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_NUM = 8;

template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        blockLength_ = totalLength / AscendC::GetBlockNum();
        tileLength_ = blockLength_ / TILE_NUM / BUFFER_NUM;
        const uint32_t blockOffset = blockLength_ * AscendC::GetBlockIdx();

        xGm_.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ T *)z + blockOffset, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
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

    __aicore__ inline void Compute()
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
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> zGm_;
    uint32_t blockLength_;
    uint32_t tileLength_;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelSubCustomTemplate<half> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelSubCustomTemplate<float> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    }
}
