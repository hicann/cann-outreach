#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_DIM = 8;


class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->blockLength = totalLength / BLOCK_DIM;
        this->tileNum = tileNum;

        this->tileLength = this->blockLength / tileNum;


        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X*)x +
            AscendC::GetBlockIdx() * blockLength,
            blockLength);


        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y*)y +
            AscendC::GetBlockIdx() * blockLength,
            blockLength);


        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(DTYPE_X));


        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            tileLength * sizeof(DTYPE_Y));
    }


    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < tileNum; i++) {

            CopyIn(i);

            Compute(i);

            CopyOut(i);
        }
    }


private:

    __aicore__ inline void CopyIn(int32_t progress)
    {
        auto xLocal = inQueueX.AllocTensor<DTYPE_X>();


        uint32_t offset =
            progress * tileLength;


        DataCopy(
            xLocal,
            xGm[offset],
            tileLength);


        inQueueX.EnQue(xLocal);
    }


    __aicore__ inline void Compute(int32_t progress)
    {
        auto xLocal =
            inQueueX.DeQue<DTYPE_X>();


        auto yLocal =
            outQueueY.AllocTensor<DTYPE_Y>();


        AscendC::Tanh(
            yLocal,
            xLocal,
            tileLength);


        outQueueY.EnQue(yLocal);


        inQueueX.FreeTensor(xLocal);
    }



    __aicore__ inline void CopyOut(int32_t progress)
    {
        auto yLocal =
            outQueueY.DeQue<DTYPE_Y>();


        uint32_t offset =
            progress * tileLength;


        DataCopy(
            yGm[offset],
            yLocal,
            tileLength);


        outQueueY.FreeTensor(yLocal);
    }



private:

    AscendC::TPipe pipe;


    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueX;


    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> outQueueY;



    AscendC::GlobalTensor<DTYPE_X> xGm;

    AscendC::GlobalTensor<DTYPE_Y> yGm;


    uint32_t blockLength;

    uint32_t tileNum;

    uint32_t tileLength;
};



extern "C" __global__ __aicore__ void tanh_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{

    REGISTER_TILING_DEFAULT(TanhCustomTilingData);


    GET_TILING_DATA(
        tilingData,
        tiling);


    KernelTanh op;


    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum);


    op.Process();
}