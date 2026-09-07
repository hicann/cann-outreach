#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

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
        // 每个AI Core处理的元素数量
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        // 每个AI Core的基本分块数量
        this->tileNum = tileNum;

        // 使用双缓冲，因此实际循环次数是tileNum * BUFFER_NUM
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 设置当前AI Core对应的输入GM地址
        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X*)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 设置当前AI Core对应的输出GM地址
        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y*)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 初始化输入、输出双缓冲队列
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DTYPE_Y));

        // 初始化三个计算临时Buffer
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
        // 双缓冲下实际需要处理tileNum * 2次
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
        // 从输入队列申请LocalTensor
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueueX.AllocTensor<DTYPE_X>();

        // 将当前Tile从GM复制到Local Memory
        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        // 将输入Tensor放入VECIN队列
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueueX.DeQue<DTYPE_X>();

        // 从输出队列申请结果Tensor
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueueY.AllocTensor<DTYPE_Y>();

        // 获取三个临时计算Tensor
        AscendC::LocalTensor<DTYPE_X> tmpLocal0 =
            tmpBuf0.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmpLocal1 =
            tmpBuf1.Get<DTYPE_X>();

        AscendC::LocalTensor<DTYPE_X> tmpLocal2 =
            tmpBuf2.Get<DTYPE_X>();

        /*
         * tanh(x) =
         *     (exp(x) - exp(-x))
         *     ------------------
         *     (exp(x) + exp(-x))
         */

        // tmpLocal2 = -x
        AscendC::Muls(
            tmpLocal2,
            xLocal,
            static_cast<DTYPE_X>(-1.0),
            this->tileLength);

        // tmpLocal0 = exp(x)
        AscendC::Exp(
            tmpLocal0,
            xLocal,
            this->tileLength);

        // tmpLocal1 = exp(-x)
        AscendC::Exp(
            tmpLocal1,
            tmpLocal2,
            this->tileLength);

        // tmpLocal2 = exp(x) - exp(-x)，作为分子
        AscendC::Sub(
            tmpLocal2,
            tmpLocal0,
            tmpLocal1,
            this->tileLength);

        // tmpLocal0 = exp(x) + exp(-x)，作为分母
        AscendC::Add(
            tmpLocal0,
            tmpLocal0,
            tmpLocal1,
            this->tileLength);

        // yLocal = 分子 / 分母
        AscendC::Div(
            yLocal,
            tmpLocal2,
            tmpLocal0,
            this->tileLength);

        // 将计算结果放入输出队列
        outQueueY.EnQue<DTYPE_Y>(yLocal);

        // 释放输入LocalTensor
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出计算结果
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueueY.DeQue<DTYPE_Y>();

        // 将结果从Local Memory复制回GM
        AscendC::DataCopy(
            yGm[progress * this->tileLength],
            yLocal,
            this->tileLength);

        // 释放输出LocalTensor
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

    KernelTanh op;

    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum);

    op.Process();
}