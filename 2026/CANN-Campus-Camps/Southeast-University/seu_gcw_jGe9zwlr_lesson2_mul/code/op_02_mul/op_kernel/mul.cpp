#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;

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
        // 获取当前 Core 编号
        uint32_t blockIdx =
            AscendC::GetBlockIdx();

        // 获取 Core 数量
        uint32_t blockNum =
            AscendC::GetBlockNum();

        // 每个 Core 负责的数据量
        blockLength =
            totalLength / blockNum;

        // 每个 Core 的 tile 数量
        this->tileNum = tileNum;

        // 每个 tile 的数据量
        tileLength =
            blockLength / tileNum / BUFFER_NUM;

        // 设置 GM 地址
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x +
            blockIdx * blockLength,
            blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y +
            blockIdx * blockLength,
            blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z +
            blockIdx * blockLength,
            blockLength);

        // 初始化输入 Queue
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        // 初始化输出 Queue
        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount =
            tileNum * BUFFER_NUM;

        for (uint32_t i = 0; i < loopCount; i++) {

            CopyIn(i);

            Compute(i);

            CopyOut(i);
        }
    }

private:

    // GM -> Local Buffer
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * tileLength],
            tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * tileLength],
            tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // Vector Mul
    __aicore__ inline void Compute(int32_t progress)
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
            tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // Local Buffer -> GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm[progress * tileLength],
            zLocal,
            tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:

    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> inQueueY;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> outQueueZ;

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