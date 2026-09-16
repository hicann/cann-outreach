#include "div_custom_template_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t DTYPE_FLOAT32 = 1;
}

template <typename T>
class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength, uint32_t tileLength)
    {
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        const uint32_t blockOffset = GetBlockIdx() * blockLength;
        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + blockOffset, blockLength);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + blockOffset, blockLength);
        zGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z) + blockOffset, blockLength);

        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < blockLength; offset += tileLength) {
            const uint32_t currentOffset = static_cast<uint32_t>(offset);
            const uint32_t length = blockLength - currentOffset < tileLength ? blockLength - currentOffset
                                                                              : tileLength;
            CopyIn(currentOffset, length);
            Compute(length);
            CopyOut(currentOffset, length);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        DataCopy(xLocal, xGlobal[offset], length);
        DataCopy(yLocal, yGlobal[offset], length);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> xLocal = xQueue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
        Div(zLocal, xLocal, yLocal, length);
        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> zLocal = zQueue.DeQue<T>();
        DataCopy(zGlobal[offset], zLocal, length);
        zQueue.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    GlobalTensor<T> zGlobal;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dataType == DTYPE_FLOAT32) {
        KernelDivCustomTemplate<float> op;
        op.Init(x, y, z, tilingData.blockLength, tilingData.tileLength);
        op.Process();
    } else {
        KernelDivCustomTemplate<half> op;
        op.Init(x, y, z, tilingData.blockLength, tilingData.tileLength);
        op.Process();
    }
}
