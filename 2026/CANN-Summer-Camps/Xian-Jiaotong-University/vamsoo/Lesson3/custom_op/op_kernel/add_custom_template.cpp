#include "kernel_operator.h"
#include "add_custom_template_tiling.h"


constexpr int32_t BUFFER_NUM = 2;


template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {

public:

    __aicore__ inline KernelAdd()
    {
    }



    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {

        uint32_t blockIdx =
            AscendC::GetBlockIdx();


        uint32_t blockNum =
            AscendC::GetBlockNum();



        // 每个core负责的数据量
        this->blockLength =
            totalLength / blockNum;



        this->tileNum =
            tileNum;



        // 每个tile大小
        this->tileLength =
            this->blockLength /
            tileNum /
            BUFFER_NUM;



        xGm.SetGlobalBuffer(
            (__gm__ dtypeX *)x +
            blockIdx * blockLength,
            blockLength);



        yGm.SetGlobalBuffer(
            (__gm__ dtypeY *)y +
            blockIdx * blockLength,
            blockLength);



        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ *)z +
            blockIdx * blockLength,
            blockLength);



        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(dtypeX));



        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            tileLength * sizeof(dtypeY));



        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            tileLength * sizeof(dtypeZ));

    }





    __aicore__ inline void Process()
    {

        int32_t loopCount =
            tileNum * BUFFER_NUM;


        for(int32_t i = 0;
            i < loopCount;
            i++)
        {

            CopyIn(i);

            Compute(i);

            CopyOut(i);

        }

    }





private:


    __aicore__ inline void CopyIn(
        int32_t progress)
    {


        auto xLocal =
            inQueueX.AllocTensor<dtypeX>();


        auto yLocal =
            inQueueY.AllocTensor<dtypeY>();



        uint32_t offset =
            progress *
            tileLength;



        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            tileLength);



        AscendC::DataCopy(
            yLocal,
            yGm[offset],
            tileLength);



        inQueueX.EnQue(
            xLocal);



        inQueueY.EnQue(
            yLocal);

    }





    __aicore__ inline void Compute(
        int32_t progress)
    {


        auto xLocal =
            inQueueX.DeQue<dtypeX>();


        auto yLocal =
            inQueueY.DeQue<dtypeY>();



        auto zLocal =
            outQueueZ.AllocTensor<dtypeZ>();



        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            tileLength);



        outQueueZ.EnQue(
            zLocal);



        inQueueX.FreeTensor(
            xLocal);



        inQueueY.FreeTensor(
            yLocal);

    }





    __aicore__ inline void CopyOut(
        int32_t progress)
    {


        auto zLocal =
            outQueueZ.DeQue<dtypeZ>();



        uint32_t offset =
            progress *
            tileLength;



        AscendC::DataCopy(
            zGm[offset],
            zLocal,
            tileLength);



        outQueueZ.FreeTensor(
            zLocal);

    }





private:


    AscendC::TPipe pipe;



    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueX;



    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueY;



    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> outQueueZ;



    AscendC::GlobalTensor<dtypeX> xGm;

    AscendC::GlobalTensor<dtypeY> yGm;

    AscendC::GlobalTensor<dtypeZ> zGm;



    uint32_t blockLength;

    uint32_t tileNum;

    uint32_t tileLength;

};





extern "C"
__global__
__aicore__
void add_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{


    REGISTER_TILING_DEFAULT(
        AddCustomTemplateTilingData);



    GET_TILING_DATA_WITH_STRUCT(
        AddCustomTemplateTilingData,
        tilingData,
        tiling);



    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;



    op.Init(
        x,
        y,
        z,
        tilingData.totalLength,
        tilingData.tileNum);



    op.Process();

}