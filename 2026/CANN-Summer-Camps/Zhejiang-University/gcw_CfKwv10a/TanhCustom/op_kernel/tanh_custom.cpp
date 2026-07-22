#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength =
        totalLength / AscendC::GetBlockNum();

        this->tileNum = tileNum;

        this->tileLength =
        this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer(
        (__gm__ DTYPE_X*)x
            + this->blockLength * AscendC::GetBlockIdx(),
        this->blockLength
    );
        yGm.SetGlobalBuffer(
        (__gm__ DTYPE_Y*)y
            + this->blockLength * AscendC::GetBlockIdx(),
        this->blockLength
    );

     pipe.InitBuffer(
        inQueueX,
        BUFFER_NUM,
        this->tileLength * sizeof(DTYPE_X)
    );
    pipe.InitBuffer(
        outQueueY,
        BUFFER_NUM,
        this->tileLength * sizeof(DTYPE_Y)
    );
     pipe.InitBuffer(
        tmpBuf0,
        this->tileLength * sizeof(DTYPE_X)
    );

    pipe.InitBuffer(
        tmpBuf1,
        this->tileLength * sizeof(DTYPE_X)
    );

    pipe.InitBuffer(
        tmpBuf2,
        this->tileLength * sizeof(DTYPE_X)
    );
    }
    __aicore__ inline void Process()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;

    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
        inQueueX.AllocTensor<DTYPE_X>();

    // GM → Local Memory
    AscendC::DataCopy(
        xLocal,
        xGm[progress * this->tileLength],
        this->tileLength
    );

    // 将已经装好数据的 LocalTensor 放入输入队列
    inQueueX.EnQue<DTYPE_X>(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
         // 从输入队列取出当前 tile
    AscendC::LocalTensor<DTYPE_X> xLocal =
        inQueueX.DeQue<DTYPE_X>();

    // 从输出队列申请空间
    AscendC::LocalTensor<DTYPE_Y> yLocal =
        outQueueY.AllocTensor<DTYPE_Y>();

    // 获取三个计算临时缓冲区
    AscendC::LocalTensor<DTYPE_X> tmpTensor0 =
        tmpBuf0.Get<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_X> tmpTensor1 =
        tmpBuf1.Get<DTYPE_X>();

    AscendC::LocalTensor<DTYPE_X> tmpTensor2 =
        tmpBuf2.Get<DTYPE_X>();

    DTYPE_X negativeOne = -1;

    // tmpTensor2 = -x
    AscendC::Muls(
        tmpTensor2,
        xLocal,
        negativeOne,
        this->tileLength
    );

    // tmpTensor0 = exp(x)
    AscendC::Exp(
        tmpTensor0,
        xLocal,
        this->tileLength
    );

    // tmpTensor1 = exp(-x)
    AscendC::Exp(
        tmpTensor1,
        tmpTensor2,
        this->tileLength
    );

    // tmpTensor2 = exp(x) - exp(-x)
    AscendC::Sub(
        tmpTensor2,
        tmpTensor0,
        tmpTensor1,
        this->tileLength
    );

    // tmpTensor0 = exp(x) + exp(-x)
    AscendC::Add(
        tmpTensor0,
        tmpTensor0,
        tmpTensor1,
        this->tileLength
    );

    // y = 分子 / 分母
    AscendC::Div(
        yLocal,
        tmpTensor2,
        tmpTensor0,
        this->tileLength
    );

    // 将结果放入输出队列
    outQueueY.EnQue<DTYPE_Y>(yLocal);

    // 释放输入 LocalTensor
    inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出计算结果
    AscendC::LocalTensor<DTYPE_Y> yLocal =
        outQueueY.DeQue<DTYPE_Y>();

    // Local Memory → GM
    AscendC::DataCopy(
        yGm[progress * this->tileLength],
        yLocal,
        this->tileLength
    );

    // 释放输出 LocalTensor
    outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: 根据tilingData计算tileNum和tileLength
    KernelTanh op;

    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum
    );

    op.Process();
}