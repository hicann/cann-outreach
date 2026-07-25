#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"


using namespace AscendC;


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



    constexpr uint32_t BUFFER_NUM = 2;

    constexpr uint32_t TILE_LENGTH = 512;



    class KernelSub
    {

    public:


        __aicore__ inline void Init(
            GM_ADDR x,
            GM_ADDR y,
            GM_ADDR z,
            uint32_t totalSize)
        {


            uint32_t blockId =
                GetBlockIdx();


            uint32_t blockNum =
                GetBlockNum();



            uint32_t baseLength =
                totalSize / blockNum;



            if(blockId == blockNum-1)
            {

                blockLength =
                    totalSize -
                    baseLength*(blockNum-1);

            }
            else
            {

                blockLength =
                    baseLength;

            }




            uint32_t offset;


            if(blockId == blockNum-1)
            {

                offset =
                    baseLength*(blockNum-1);

            }
            else
            {

                offset =
                    baseLength*blockId;

            }





            xGm.SetGlobalBuffer(
                (__gm__ half*)x + offset,
                blockLength
            );


            yGm.SetGlobalBuffer(
                (__gm__ half*)y + offset,
                blockLength
            );


            zGm.SetGlobalBuffer(
                (__gm__ half*)z + offset,
                blockLength
            );





            tileNum =
                (blockLength + TILE_LENGTH -1)
                / TILE_LENGTH;




            pipe.InitBuffer(
                inQueueX,
                BUFFER_NUM,
                TILE_LENGTH*sizeof(half)
            );


            pipe.InitBuffer(
                inQueueY,
                BUFFER_NUM,
                TILE_LENGTH*sizeof(half)
            );


            pipe.InitBuffer(
                outQueueZ,
                BUFFER_NUM,
                TILE_LENGTH*sizeof(half)
            );


        }




        __aicore__ inline void Process()
        {


            for(uint32_t i=0;i<tileNum;i++)
            {


                uint32_t length =
                    TILE_LENGTH;



                if(i==tileNum-1)
                {

                    length =
                        blockLength -
                        i*TILE_LENGTH;

                }



                CopyIn(i,length);


                Compute(length);


                CopyOut(i,length);


            }


        }





        __aicore__ inline void CopyIn(
            uint32_t progress,
            uint32_t length)
        {


            LocalTensor<half> xLocal =
                inQueueX.AllocTensor<half>();


            LocalTensor<half> yLocal =
                inQueueY.AllocTensor<half>();




            DataCopy(
                xLocal,
                xGm[progress*TILE_LENGTH],
                length
            );


            DataCopy(
                yLocal,
                yGm[progress*TILE_LENGTH],
                length
            );



            inQueueX.EnQue(xLocal);

            inQueueY.EnQue(yLocal);


        }





        __aicore__ inline void Compute(
            uint32_t length)
        {


            LocalTensor<half> xLocal =
                inQueueX.DeQue<half>();


            LocalTensor<half> yLocal =
                inQueueY.DeQue<half>();



            LocalTensor<half> zLocal =
                outQueueZ.AllocTensor<half>();




            Sub(
                zLocal,
                xLocal,
                yLocal,
                length
            );



            outQueueZ.EnQue(zLocal);



            inQueueX.FreeTensor(xLocal);

            inQueueY.FreeTensor(yLocal);



        }





        __aicore__ inline void CopyOut(
            uint32_t progress,
            uint32_t length)
        {


            LocalTensor<half> zLocal =
                outQueueZ.DeQue<half>();



            DataCopy(
                zGm[progress*TILE_LENGTH],
                zLocal,
                length
            );


            outQueueZ.FreeTensor(zLocal);



        }





    private:


        AscendC::TPipe pipe;



        TQue<QuePosition::VECIN, BUFFER_NUM>
            inQueueX;


        TQue<QuePosition::VECIN, BUFFER_NUM>
            inQueueY;


        TQue<QuePosition::VECOUT, BUFFER_NUM>
            outQueueZ;



        GlobalTensor<half> xGm;

        GlobalTensor<half> yGm;

        GlobalTensor<half> zGm;



        uint32_t blockLength;

        uint32_t tileNum;


    };




    KernelSub op;


    op.Init(
        x,
        y,
        z,
        tilingData.size
    );


    op.Process();

}