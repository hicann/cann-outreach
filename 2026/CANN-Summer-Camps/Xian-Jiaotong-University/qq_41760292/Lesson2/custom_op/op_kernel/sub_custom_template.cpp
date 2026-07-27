#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t QUEUE_DEPTH = 1;

class KernelSubOp {
public:
    __aicore__ inline KernelSubOp() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t elementCount)
    {
        const uint32_t coreNum = GetBlockNum();
        const uint32_t coreId = GetBlockIdx();

        ASSERT(coreNum > 0 && "invalid block num");

        uint32_t avgNum = elementCount / coreNum;
        uint32_t remain = elementCount % coreNum;

        blockSize = avgNum;
        if (coreId < remain) {
            blockSize++;
        }

        offset = avgNum * coreId + (coreId < remain ? coreId : remain);

        if (blockSize == 0) {
            return;
        }

        xTensor.SetGlobalBuffer((__gm__ DTYPE_X*)x + offset, blockSize);
        yTensor.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, blockSize);
        zTensor.SetGlobalBuffer((__gm__ DTYPE_Z*)z + offset, blockSize);

        pipe.InitBuffer(xQueue, QUEUE_DEPTH, sizeof(DTYPE_X) * blockSize);
        pipe.InitBuffer(yQueue, QUEUE_DEPTH, sizeof(DTYPE_Y) * blockSize);
        pipe.InitBuffer(zQueue, QUEUE_DEPTH, sizeof(DTYPE_Z) * blockSize);
    }

    __aicore__ inline void Process()
    {
        if (blockSize == 0) {
            return;
        }

        LoadInput();
        DoSub();
        StoreResult();
    }

private:
    __aicore__ inline void LoadInput()
    {
        auto xLocal = xQueue.AllocTensor<DTYPE_X>();
        auto yLocal = yQueue.AllocTensor<DTYPE_Y>();

        DataCopy(xLocal, xTensor[0], blockSize);
        DataCopy(yLocal, yTensor[0], blockSize);

        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void DoSub()
    {
        auto xLocal = xQueue.DeQue<DTYPE_X>();
        auto yLocal = yQueue.DeQue<DTYPE_Y>();
        auto zLocal = zQueue.AllocTensor<DTYPE_Z>();

        Sub(zLocal, xLocal, yLocal, blockSize);

        zQueue.EnQue(zLocal);

        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void StoreResult()
    {
        auto zLocal = zQueue.DeQue<DTYPE_Z>();

        DataCopy(zTensor[0], zLocal, blockSize);

        zQueue.FreeTensor(zLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, QUEUE_DEPTH> xQueue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> yQueue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> zQueue;

    GlobalTensor<DTYPE_X> xTensor;
    GlobalTensor<DTYPE_Y> yTensor;
    GlobalTensor<DTYPE_Z> zTensor;

    uint32_t blockSize = 0;
    uint32_t offset = 0;
};

extern "C" __global__ __aicore__
void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    (void)workspace;

    KernelSubOp kernel;
    kernel.Init(x, y, z, tilingData.size);
    kernel.Process();
}