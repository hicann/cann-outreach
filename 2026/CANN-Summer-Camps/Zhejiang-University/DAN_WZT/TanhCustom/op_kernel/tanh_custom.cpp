#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // double buffer

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 每个核需要处理的元素数量
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        // 每个核在双缓冲之前的逻辑切块数
        this->tileNum = tileNum;

        // 开启双缓冲后，每次循环实际处理的元素数量
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        // 根据当前核编号，让每个核指向自己负责的 GM 区间
        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X*)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y*)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 为输入、输出队列分配双缓冲
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_Y));

        // 为 tanh 计算过程中的三个中间 Tensor 分配 UB
        pipe.InitBuffer(
            tmpBuf0,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            tmpBuf1,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            tmpBuf2,
            this->tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        // 因为开启了双缓冲，实际微块数量是 tileNum * 2
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

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // Compute 阶段不需要 progress，队列保证顺序对应
        (void)progress;

        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueueX.DeQue<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> tmp0 =
            tmpBuf0.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmp1 =
            tmpBuf1.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmp2 =
            tmpBuf2.Get<DTYPE_X>();

        /*
         * tanh(x) = (exp(x) - exp(-x))
         *           -------------------
         *           (exp(x) + exp(-x))
         *
         * 中间变量安排：
         * tmp2 = -x
         * tmp0 = exp(x)
         * tmp1 = exp(-x)
         * tmp2 = exp(x) - exp(-x)
         * tmp1 = exp(x) + exp(-x)
         * y     = tmp2 / tmp1
         */

        AscendC::Muls(
            tmp2,
            xLocal,
            static_cast<DTYPE_X>(-1.0f),
            this->tileLength);

        AscendC::Exp(
            tmp0,
            xLocal,
            this->tileLength);

        AscendC::Exp(
            tmp1,
            tmp2,
            this->tileLength);

        AscendC::Sub(
            tmp2,
            tmp0,
            tmp1,
            this->tileLength);

        AscendC::Add(
            tmp1,
            tmp0,
            tmp1,
            this->tileLength);

        AscendC::Div(
            yLocal,
            tmp2,
            tmp1,
            this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueueY.DeQue<DTYPE_Y>();

        AscendC::DataCopy(
            yGm[progress * this->tileLength],
            yLocal,
            this->tileLength);

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

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> tmpBuf0;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> tmpBuf1;

    AscendC::TBuf<
        AscendC::QuePosition::VECCALC> tmpBuf2;

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
    GET_TILING_DATA(tilingData, tiling);

    // 当前实现不需要 GM workspace
    (void)workspace;

    KernelTanh op;
    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum);

    op.Process();
}