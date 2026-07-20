#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;


constexpr int32_t BUFFER_NUM = 1;


class SubKernel
{
public:

    __aicore__ inline SubKernel()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalSize)
    {

        uint32_t coreNum = GetBlockNum();
        uint32_t coreId = GetBlockIdx();


        uint32_t avg = totalSize / coreNum;
        uint32_t extra = totalSize % coreNum;


        length = avg;

        if(coreId < extra)
        {
            length++;
        }


        offset = coreId * avg;

        if(coreId < extra)
        {
            offset += coreId;
        }
        else
        {
            offset += extra;
        }


        if(length == 0)
        {
            return;
        }


        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X*)x + offset,
            length
        );

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y*)y + offset,
            length
        );

        zGm.SetGlobalBuffer(
            (__gm__ DTYPE_Z*)z + offset,
            length
        );


        pipe.InitBuffer(
            xQueue,
            BUFFER_NUM,
            length*sizeof(DTYPE_X)
        );

        pipe.InitBuffer(
            yQueue,
            BUFFER_NUM,
            length*sizeof(DTYPE_Y)
        );

        pipe.InitBuffer(
            zQueue,
            BUFFER_NUM,
            length*sizeof(DTYPE_Z)
        );

    }



    __aicore__ inline void Run()
    {

        if(length==0)
            return;


        CopyIn();

        Calculate();

        CopyOut();

    }



private:


    __aicore__ inline void CopyIn()
    {

        auto xLocal =
            xQueue.AllocTensor<DTYPE_X>();

        auto yLocal =
            yQueue.AllocTensor<DTYPE_Y>();


        DataCopy(
            xLocal,
            xGm,
            length
        );


        DataCopy(
            yLocal,
            yGm,
            length
        );


        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);

    }



    __aicore__ inline void Calculate()
    {

        auto xLocal =
            xQueue.DeQue<DTYPE_X>();

        auto yLocal =
            yQueue.DeQue<DTYPE_Y>();


        auto zLocal =
            zQueue.AllocTensor<DTYPE_Z>();


        Sub(
            zLocal,
            xLocal,
            yLocal,
            length
        );


        zQueue.EnQue(zLocal);


        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);

    }



    __aicore__ inline void CopyOut()
    {

        auto zLocal =
            zQueue.DeQue<DTYPE_Z>();


        DataCopy(
            zGm,
            zLocal,
            length
        );


        zQueue.FreeTensor(zLocal);

    }



private:

    TPipe pipe;


    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;


    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;


    uint32_t length=0;
    uint32_t offset=0;

};




extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{

    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);

    GET_TILING_DATA(tilingData, tiling);


    (void)workspace;


    SubKernel op;


    op.Init(
        x,
        y,
        z,
        tilingData.size
    );


    op.Run();

}