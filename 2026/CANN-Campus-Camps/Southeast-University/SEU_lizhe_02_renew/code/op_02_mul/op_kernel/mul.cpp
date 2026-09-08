// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 每个核负责的数据量
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        this->tileNum = tileNum;

        // 双缓冲，因此每个 tile 再除以 BUFFER_NUM
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 设置当前核负责的 GM 地址范围
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 初始化双缓冲队列
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
    }


    __aicore__ inline void Process()
    {
        // 每个 tile 使用双缓冲
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
        // 从 UB 队列申请 LocalTensor
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // GM -> UB
        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        // 入队
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    __aicore__ inline void Compute(int32_t progress)
    {
        // 输入出队
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 为结果申请 LocalTensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 核心计算：
        // z = x * y
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 结果入队
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 结果出队
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // UB -> GM
        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        // 释放结果 Tensor
        outQueueZ.FreeTensor(zLocal);
    }


private:

    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.tileNum);

    op.Process();
}