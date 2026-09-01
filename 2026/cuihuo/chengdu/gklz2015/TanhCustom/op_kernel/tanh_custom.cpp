#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using namespace AscendC;


constexpr int32_t BUFFER_NUM = 2;


class KernelTanh {

public:

    __aicore__ inline KernelTanh()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {

        this->totalLength = totalLength;
        this->tileNum = tileNum;


        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(x),
            totalLength
        );


        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(y),
            totalLength
        );


        tileLength = totalLength / tileNum;



        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(half)
        );


        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            tileLength * sizeof(half)
        );


        pipe.InitBuffer(
            xFloatBuf,
            tileLength * sizeof(float)
        );


        pipe.InitBuffer(
            negXBuf,
            tileLength * sizeof(float)
        );


        pipe.InitBuffer(
            expXBuf,
            tileLength * sizeof(float)
        );


        pipe.InitBuffer(
            expNegXBuf,
            tileLength * sizeof(float)
        );


        pipe.InitBuffer(
            denominatorBuf,
            tileLength * sizeof(float)
        );

    }



    __aicore__ inline void Process()
    {

        for(uint32_t i = 0; i < tileNum; i++)
        {

            CopyIn(i);

            Compute();

            CopyOut(i);

        }

    }



private:


    __aicore__ inline void CopyIn(
        uint32_t progress)
    {

        LocalTensor<half> xLocal =
            inQueueX.AllocTensor<half>();


        DataCopy(
            xLocal,
            xGm[progress * tileLength],
            tileLength
        );


        inQueueX.EnQue(xLocal);

    }




    __aicore__ inline void Compute()
    {


        LocalTensor<half> xLocal =
            inQueueX.DeQue<half>();


        LocalTensor<half> yLocal =
            outQueueY.AllocTensor<half>();



        LocalTensor<float> xFloat =
            xFloatBuf.Get<float>();


        LocalTensor<float> negX =
            negXBuf.Get<float>();


        LocalTensor<float> expX =
            expXBuf.Get<float>();


        LocalTensor<float> expNegX =
            expNegXBuf.Get<float>();


        LocalTensor<float> denominator =
            denominatorBuf.Get<float>();



        /*
         * half -> float
         */
        Cast(
            xFloat,
            xLocal,
            RoundMode::CAST_NONE,
            tileLength
        );



        /*
         * exp(x)
         */
        Exp(
            expX,
            xFloat,
            tileLength
        );



        /*
         * -x
         */
        Muls(
            negX,
            xFloat,
            -1.0f,
            tileLength
        );



        /*
         * exp(-x)
         */
        Exp(
            expNegX,
            negX,
            tileLength
        );



        /*
         * denominator =
         *
         * exp(x)+exp(-x)
         *
         */
        Add(
            denominator,
            expX,
            expNegX,
            tileLength
        );



        /*
         * numerator =
         *
         * exp(x)-exp(-x)
         *
         */
        Sub(
            expX,
            expX,
            expNegX,
            tileLength
        );



        /*
         *
         * tanh =
         *
         * numerator / denominator
         *
         */
        Div(
            expX,
            expX,
            denominator,
            tileLength
        );



        /*
         * float -> half
         */
        Cast(
            yLocal,
            expX,
            RoundMode::CAST_NONE,
            tileLength
        );



        outQueueY.EnQue(yLocal);


        inQueueX.FreeTensor(xLocal);

    }




    __aicore__ inline void CopyOut(
        uint32_t progress)
    {

        LocalTensor<half> yLocal =
            outQueueY.DeQue<half>();


        DataCopy(
            yGm[progress * tileLength],
            yLocal,
            tileLength
        );


        outQueueY.FreeTensor(yLocal);

    }



private:


    TPipe pipe;



    TQue<QuePosition::VECIN, BUFFER_NUM>
        inQueueX;


    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outQueueY;



    TBuf<TPosition::VECCALC>
        xFloatBuf;


    TBuf<TPosition::VECCALC>
        negXBuf;


    TBuf<TPosition::VECCALC>
        expXBuf;


    TBuf<TPosition::VECCALC>
        expNegXBuf;


    TBuf<TPosition::VECCALC>
        denominatorBuf;



    GlobalTensor<half> xGm;

    GlobalTensor<half> yGm;



    uint32_t totalLength;

    uint32_t tileNum;

    uint32_t tileLength;

};




extern "C" __global__
__aicore__
void tanh_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{


    REGISTER_TILING_DEFAULT(TanhCustomTilingData);


    GET_TILING_DATA(
        tilingData,
        tiling
    );



    KernelTanh op;



    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum
    );



    op.Process();

}
