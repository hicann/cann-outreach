#include "kernel_operator.h"

#include "add_custom_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kFloat16BlockLength = 16;

}  // namespace

class KernelAddCustom {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileLength)
    {
        xGm_.SetGlobalBuffer((__gm__ half*)x, totalLength);
        yGm_.SetGlobalBuffer((__gm__ half*)y, totalLength);
        zGm_.SetGlobalBuffer((__gm__ half*)z, totalLength);
        totalLength_ = totalLength;
        tileLength_ = tileLength;

        pipe_.InitBuffer(inQueueX_, kBufferNum, tileLength_ * sizeof(half));
        pipe_.InitBuffer(inQueueY_, kBufferNum, tileLength_ * sizeof(half));
        pipe_.InitBuffer(outQueueZ_, kBufferNum, tileLength_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < totalLength_;) {
            const uint32_t count = (totalLength_ - offset > tileLength_)
                ? tileLength_
                : totalLength_ - offset;
            const uint32_t alignedCount =
                (count + kFloat16BlockLength - 1) / kFloat16BlockLength * kFloat16BlockLength;

            CopyIn(offset, count, alignedCount);
            Compute(alignedCount);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count, uint32_t alignedCount)
    {
        LocalTensor<half> xLocal = inQueueX_.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY_.AllocTensor<half>();
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        const DataCopyPadExtParams<half> padParams{
            true, 0, static_cast<uint8_t>(alignedCount - count), 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<half> xLocal = inQueueX_.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY_.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ_.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, count);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<half> zLocal = outQueueZ_.DeQue<half>();
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, kBufferNum> inQueueX_;
    TQue<TPosition::VECIN, kBufferNum> inQueueY_;
    TQue<TPosition::VECOUT, kBufferNum> outQueueZ_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> zGm_;
    uint32_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}
