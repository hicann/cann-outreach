#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_DEPTH = 2;
constexpr int32_t PER_TILE_ELEM_COUNT = 512;

class KernelSubOp {
public:
    __aicore__ inline KernelSubOp() {}
    __aicore__ inline void Init(GM_ADDR xGmBase, GM_ADDR yGmBase, GM_ADDR zGmBase, uint32_t totalElemNum)
    {
        ASSERT(GetBlockNum() != 0 && totalElemNum != 0);
        uint32_t coreCount = GetBlockNum();
        uint32_t elemPerCore = (totalElemNum + coreCount - 1) / coreCount;
        uint32_t coreStartOffset = GetBlockIdx() * elemPerCore;

        if (coreStartOffset + elemPerCore > totalElemNum) {
            elemPerCore = totalElemNum - coreStartOffset;
        }

        singleCoreElemNum = elemPerCore;
        tileElemCount = PER_TILE_ELEM_COUNT;
        tileLoopTimes = (singleCoreElemNum + tileElemCount - 1) / tileElemCount;

        xGlobal.SetGlobalBuffer((__gm__ half *)xGmBase + coreStartOffset, singleCoreElemNum);
        yGlobal.SetGlobalBuffer((__gm__ half *)yGmBase + coreStartOffset, singleCoreElemNum);
        zGlobal.SetGlobalBuffer((__gm__ half *)zGmBase + coreStartOffset, singleCoreElemNum);

        dataPipe.InitBuffer(inputQueueX, DOUBLE_BUFFER_DEPTH, tileElemCount * sizeof(half));
        dataPipe.InitBuffer(inputQueueY, DOUBLE_BUFFER_DEPTH, tileElemCount * sizeof(half));
        dataPipe.InitBuffer(outputQueueZ, DOUBLE_BUFFER_DEPTH, tileElemCount * sizeof(half));
    }

    __aicore__ inline void ExecutePipeline()
    {
        for (int32_t step = 0; step < tileLoopTimes; step++)
        {
            LoadData(step);
            Calculate(step);
            StoreData(step);
        }
    }

private:
    __aicore__ inline void LoadData(int32_t stepIndex)
    {
        LocalTensor<half> xLocal = inputQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inputQueueY.AllocTensor<half>();

        uint32_t gmOffset = stepIndex * tileElemCount;
        DataCopy(xLocal, xGlobal[gmOffset], tileElemCount);
        DataCopy(yLocal, yGlobal[gmOffset], tileElemCount);

        inputQueueX.EnQue<half>(xLocal);
        inputQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void Calculate(int32_t stepIndex)
    {
        LocalTensor<half> xBuf = inputQueueX.DeQue<half>();
        LocalTensor<half> yBuf = inputQueueY.DeQue<half>();
        LocalTensor<half> zBuf = outputQueueZ.AllocTensor<half>();

        Sub(zBuf, xBuf, yBuf, tileElemCount);

        outputQueueZ.EnQue<half>(zBuf);
        inputQueueX.FreeTensor(xBuf);
        inputQueueY.FreeTensor(yBuf);
    }

    __aicore__ inline void StoreData(int32_t stepIndex)
    {
        LocalTensor<half> zLocal = outputQueueZ.DeQue<half>();
        uint32_t gmOffset = stepIndex * tileElemCount;
        DataCopy(zGlobal[gmOffset], zLocal, tileElemCount);
        outputQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe dataPipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> inputQueueX, inputQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_DEPTH> outputQueueZ;

    GlobalTensor<half> xGlobal;
    GlobalTensor<half> yGlobal;
    GlobalTensor<half> zGlobal;

    uint32_t singleCoreElemNum = 0;
    uint32_t tileLoopTimes = 0;
    uint32_t tileElemCount = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSubOp opInstance;
    opInstance.Init(x, y, z, tilingData.size);
    opInstance.ExecutePipeline();
}