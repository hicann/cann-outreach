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
template<typename Tx, typename Ty, typename Tz>
class VecSubOperator {
public:
    __aicore__ inline VecSubOperator() {}

    __aicore__ inline void Init(GM_ADDR xBase, GM_ADDR yBase, GM_ADDR zBase, uint32_t totalElem, uint32_t tilePerCore)
    {
        uint32_t coreCount = AscendC::GetBlockNum();
        uint32_t coreId = AscendC::GetBlockIdx();
        this->coreTotalElem = totalElem / coreCount;
        this->tilePerCore = tilePerCore;
        this->perTileElem = this->coreTotalElem / tilePerCore;

        xGlobalMem.SetGlobalBuffer((__gm__ Tx*)xBase + coreId * this->coreTotalElem, this->coreTotalElem);
        yGlobalMem.SetGlobalBuffer((__gm__ Ty*)yBase + coreId * this->coreTotalElem, this->coreTotalElem);
        zGlobalMem.SetGlobalBuffer((__gm__ Tz*)zBase + coreId * this->coreTotalElem, this->coreTotalElem);

        dataPipeline.InitBuffer(inputQueX, BUFFER_NUM, this->perTileElem * sizeof(Tx));
        dataPipeline.InitBuffer(inputQueY, BUFFER_NUM, this->perTileElem * sizeof(Ty));
        dataPipeline.InitBuffer(outputQueZ, BUFFER_NUM, this->perTileElem * sizeof(Tz));
    }

    __aicore__ inline void Execute()
    {
        for (int32_t stepIdx = 0; stepIdx < this->tilePerCore; stepIdx++)
        {
            LoadInput(stepIdx);
            CalculateSub(stepIdx);
            StoreOutput(stepIdx);
        }
    }

private:
    __aicore__ inline void LoadInput(int32_t stepIdx)
    {
        uint32_t offset = stepIdx * this->perTileElem;
        AscendC::LocalTensor<Tx> xBuf = inputQueX.AllocTensor<Tx>();
        AscendC::LocalTensor<Ty> yBuf = inputQueY.AllocTensor<Ty>();

        AscendC::DataCopy(xBuf, xGlobalMem[offset], this->perTileElem);
        AscendC::DataCopy(yBuf, yGlobalMem[offset], this->perTileElem);

        inputQueX.EnQue(xBuf);
        inputQueY.EnQue(yBuf);
    }

    __aicore__ inline void CalculateSub(int32_t stepIdx)
    {
        AscendC::LocalTensor<Tx> xBuf = inputQueX.DeQue<Tx>();
        AscendC::LocalTensor<Ty> yBuf = inputQueY.DeQue<Ty>();
        AscendC::LocalTensor<Tz> zBuf = outputQueZ.AllocTensor<Tz>();

        AscendC::Sub(zBuf, xBuf, yBuf, this->perTileElem);

        outputQueZ.EnQue(zBuf);
        inputQueX.FreeTensor(xBuf);
        inputQueY.FreeTensor(yBuf);
    }

    __aicore__ inline void StoreOutput(int32_t stepIdx)
    {
        uint32_t offset = stepIdx * this->perTileElem;
        AscendC::LocalTensor<Tz> zBuf = outputQueZ.DeQue<Tz>();
        AscendC::DataCopy(zGlobalMem[offset], zBuf, this->perTileElem);
        outputQueZ.FreeTensor(zBuf);
    }

private:
    uint32_t coreTotalElem;
    uint32_t perTileElem;
    uint32_t tilePerCore;

    AscendC::TPipe dataPipeline;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueZ;

    AscendC::GlobalTensor<Tx> xGlobalMem;
    AscendC::GlobalTensor<Ty> yGlobalMem;
    AscendC::GlobalTensor<Tz> zGlobalMem;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingCfg, tiling);

    VecSubOperator<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingCfg.size, 8);
    op.Execute();
}