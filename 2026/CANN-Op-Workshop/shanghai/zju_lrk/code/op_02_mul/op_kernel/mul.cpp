// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t blockLength,
        uint32_t tileLength,
        uint32_t tileNum)
    {
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->tileNum = tileNum;

        // 每个AI Core处理Global Memory中属于自己的连续数据段
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);
        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);
        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 为CopyIn / Compute / CopyOut流水线申请Local Memory
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);
        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 本题核心：逐元素乘法 z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
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
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(
        x,
        y,
        z,
        tiling_data.blockLength,
        tiling_data.tileLength,
        tiling_data.tileNum);
    op.Process();
}
