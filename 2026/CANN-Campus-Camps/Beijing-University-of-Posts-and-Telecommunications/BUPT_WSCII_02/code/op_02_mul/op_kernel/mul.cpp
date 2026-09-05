#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 每个核负责的数据长度
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        this->tileNum = tileNum;

        /*
         * 因为使用双缓冲：
         *
         * 每个逻辑 tile 被拆成 BUFFER_NUM 份
         */
        this->tileLength =
            this->blockLength /
            this->tileNum /
            BUFFER_NUM;

        uint32_t blockOffset =
            this->blockLength *
            AscendC::GetBlockIdx();

        // 每个核绑定自己负责的 GM 区域
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + blockOffset,
            this->blockLength
        );

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + blockOffset,
            this->blockLength
        );

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + blockOffset,
            this->blockLength
        );

        // UB 双缓冲
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );
    }

    __aicore__ inline void Process()
    {
        /*
         * tileNum = 1
         * BUFFER_NUM = 2
         *
         * 因此 loopCount = 2
         *
         * 每次处理 1024 elements。
         */
        int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0;
             i < loopCount;
             ++i)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:

    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        uint32_t offset =
            progress * this->tileLength;

        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            this->tileLength
        );

        AscendC::DataCopy(
            yLocal,
            yGm[offset],
            this->tileLength
        );

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // z = x * y
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength
        );

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t offset =
            progress * this->tileLength;

        AscendC::DataCopy(
            zGm[offset],
            zLocal,
            this->tileLength
        );

        outQueueZ.FreeTensor(zLocal);
    }

private:

    // Pipe
    AscendC::TPipe pipe;

    // 输入队列
    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM
    > inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM
    > inQueueY;

    // 输出队列
    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM
    > outQueueZ;

    // GM Tensor
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
        tilingData,
        tiling
    );

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tilingData.length,
        tilingData.tileNum
    );

    op.Process();
}