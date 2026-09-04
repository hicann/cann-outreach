// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t length, uint32_t tileLength, uint32_t tileNum)
    {
        this->length = length;
        this->tileLength = tileLength;
        this->tileNum = tileNum;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, length);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; ++i) {
            uint32_t offset = i * tileLength;
            uint32_t computeLength =
                (offset + tileLength <= length) ? tileLength : (length - offset);

            CopyIn(offset, computeLength);
            Compute(computeLength);
            CopyOut(offset, computeLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t computeLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm[offset], computeLength);
        DataCopy(yLocal, yGm[offset], computeLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t computeLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, computeLength);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t computeLength)
    {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        DataCopy(zGm[offset], zLocal, computeLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t length;
    uint32_t tileLength;
    uint32_t tileNum;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileLength, tiling_data.tileNum);
    op.Process();
}