#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 1;


template <typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t size)
    {
        this->size = size;

        xGm.SetGlobalBuffer(
            (__gm__ T*)x,
            size);

        yGm.SetGlobalBuffer(
            (__gm__ T*)y,
            size);

        zGm.SetGlobalBuffer(
            (__gm__ T*)z,
            size);


        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            size * sizeof(T));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            size * sizeof(T));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            size * sizeof(T));
    }


    __aicore__ inline void Process()
    {
        CopyIn();

        Compute();

        CopyOut();
    }


private:

    __aicore__ inline void CopyIn()
    {
        auto xLocal =
            inQueueX.AllocTensor<T>();

        auto yLocal =
            inQueueY.AllocTensor<T>();


        AscendC::DataCopy(
            xLocal,
            xGm,
            size);


        AscendC::DataCopy(
            yLocal,
            yGm,
            size);


        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    __aicore__ inline void Compute()
    {
        auto xLocal =
            inQueueX.DeQue<T>();

        auto yLocal =
            inQueueY.DeQue<T>();

        auto zLocal =
            outQueueZ.AllocTensor<T>();


        AscendC::Sub(
            zLocal,
            xLocal,
            yLocal,
            size);


        outQueueZ.EnQue(zLocal);


        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    __aicore__ inline void CopyOut()
    {
        auto zLocal =
            outQueueZ.DeQue<T>();


        AscendC::DataCopy(
            zGm,
            zLocal,
            size);


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


    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;


    uint32_t size;
};



extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);

    GET_TILING_DATA(
        tilingData,
        tiling);


    KernelSub<DTYPE_X> op;


    op.Init(
        x,
        y,
        z,
        tilingData.size);


    op.Process();
}