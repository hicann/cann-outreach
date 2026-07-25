#include "kernel_operator.h"
#include "add_custom_template_tiling.h"


// 双缓冲
constexpr int32_t BUFFER_NUM = 1;



template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd
{

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


        this->blockLength =
            totalLength /
            AscendC::GetBlockNum();



        this->tileNum =
            tileNum;



        /*
         * 注意:
         * tile大小不应该除BUFFER_NUM
         */
        this->tileLength = this->blockLength / tileNum;



        uint32_t offset =
            AscendC::GetBlockIdx()
            *
            this->blockLength;



        xGm.SetGlobalBuffer(
            (__gm__ dtypeX*)x + offset,
            this->blockLength);



        yGm.SetGlobalBuffer(
            (__gm__ dtypeY*)y + offset,
            this->blockLength);



        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ*)z + offset,
            this->blockLength);




        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength*sizeof(dtypeX));



        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength*sizeof(dtypeY));



        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength*sizeof(dtypeZ));

    }




   __aicore__ inline void Process()
{
    for (int32_t i = 0; i < this->tileNum; i++) {
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



        inQueueX.EnQue(xLocal);

        inQueueY.EnQue(yLocal);

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




        outQueueZ.EnQue(zLocal);



        inQueueX.FreeTensor(xLocal);

        inQueueY.FreeTensor(yLocal);

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



        outQueueZ.FreeTensor(zLocal);

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







__global__ __aicore__ void add_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{


    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);



    GET_TILING_DATA_WITH_STRUCT(
        AddCustomTemplateTilingData,
        tiling_data,
        tiling);



    KernelAdd<DTYPE_X,DTYPE_Y,DTYPE_Z> op;



    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.tileNum);



    op.Process();

}