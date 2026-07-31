#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

using namespace AscendC;

// 双缓冲性能优化
constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class KernelAddCustomTemplate {
public:
    __aicore__ inline KernelAddCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // 🔴 P0修复：用GetBlockNum()获取40核，除以40，不越界
        blockLength_ = totalLength / GetBlockNum();
        tileNum_ = tileNum;
        tileLength_ = blockLength_ / tileNum_;

        const uint32_t blockOffset = blockLength_ * GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ T*)z + blockOffset, blockLength_);

        // 双缓冲分块流水线，性能优化核心
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 分块循环，host侧传tileNum=360
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
        DataCopy(xLocal, xGm_[progress * tileLength_], tileLength_);
        DataCopy(yLocal, yGm_[progress * tileLength_], tileLength_);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
        Add(zLocal, xLocal, yLocal, tileLength_);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<T> zLocal = outQueueZ_.DeQue<T>();
        DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> zGm_;
    uint32_t blockLength_;
    uint32_t tileNum_;
    uint32_t tileLength_;
};

template <typename T>
__aicore__ inline void RunAdd(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              const AddCustomTemplateTilingData& tilingData)
{
    KernelAddCustomTemplate<T> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        RunAdd<float>(x, y, z, tilingData);
    } else {
        RunAdd<half>(x, y, z, tilingData);
    }
}