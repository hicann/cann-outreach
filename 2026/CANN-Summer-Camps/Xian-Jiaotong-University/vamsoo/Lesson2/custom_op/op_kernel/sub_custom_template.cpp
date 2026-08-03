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



class SubCustomKernel
{

public:

    __aicore__ inline
    SubCustomKernel()
    {

    }



    __aicore__ inline
    void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t size)
    {

        uint32_t blockIdx =
            AscendC::GetBlockIdx();


        // 8个核，每个核一行
        uint32_t blockLength =
            size / 8;


        this->length = blockLength;



        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_X*>(x)
            + blockIdx * blockLength,
            blockLength);



        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_Y*>(y)
            + blockIdx * blockLength,
            blockLength);



        zGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_Z*>(z)
            + blockIdx * blockLength,
            blockLength);



        pipe.InitBuffer(
            xQueue,
            1,
            blockLength * sizeof(DTYPE_X));



        pipe.InitBuffer(
            yQueue,
            1,
            blockLength * sizeof(DTYPE_Y));



        pipe.InitBuffer(
            zQueue,
            1,
            blockLength * sizeof(DTYPE_Z));

    }





    __aicore__ inline
    void Process()
    {


        /*
         * Copy In
         */

        auto xLocal =
            xQueue.AllocTensor<DTYPE_X>();


        auto yLocal =
            yQueue.AllocTensor<DTYPE_Y>();



        AscendC::DataCopy(
            xLocal,
            xGm,
            length);



        AscendC::DataCopy(
            yLocal,
            yGm,
            length);



        xQueue.EnQue(xLocal);

        yQueue.EnQue(yLocal);



        xLocal =
            xQueue.DeQue<DTYPE_X>();


        yLocal =
            yQueue.DeQue<DTYPE_Y>();



        /*
         * Compute
         */

        auto zLocal =
            zQueue.AllocTensor<DTYPE_Z>();



        AscendC::Sub(
            zLocal,
            xLocal,
            yLocal,
            length);



        zQueue.EnQue(zLocal);



        xQueue.FreeTensor(
            xLocal);


        yQueue.FreeTensor(
            yLocal);



        /*
         * Copy Out
         */


        auto outLocal =
            zQueue.DeQue<DTYPE_Z>();



        AscendC::DataCopy(
            zGm,
            outLocal,
            length);



        zQueue.FreeTensor(
            outLocal);

    }



private:


    uint32_t length;



    AscendC::TPipe pipe;



    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> xQueue;



    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> yQueue;



    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> zQueue;



    AscendC::GlobalTensor<DTYPE_X>
        xGm;



    AscendC::GlobalTensor<DTYPE_Y>
        yGm;



    AscendC::GlobalTensor<DTYPE_Z>
        zGm;

};






extern "C"
__global__
__aicore__
void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{


    REGISTER_TILING_DEFAULT(
        SubCustomTemplateTilingData);



    GET_TILING_DATA(
        tilingData,
        tiling);



    SubCustomKernel op;



    op.Init(
        x,
        y,
        z,
        tilingData.size);



    op.Process();

}