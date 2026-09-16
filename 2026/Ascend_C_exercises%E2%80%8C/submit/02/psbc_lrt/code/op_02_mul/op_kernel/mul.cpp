// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tiling) {
        totalLength    = tiling.totalLength;
        tileNum        = tiling.tileNum;
        tileLength     = tiling.tileLength;
        lastTileLength = tiling.lastTileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, totalLength);

        // 缓冲区长度取对齐块长度（至少为 1，避免空 Tensor 分配 0 字节）
        uint32_t bufLen = (tileLength > 0) ? tileLength : 1;
        // 双缓冲：队列深度为 2，让相邻块的搬运与计算重叠
        pipe.InitBuffer(inQueueX, 2, bufLen * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 2, bufLen * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 2, bufLen * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t GetCurLength(uint32_t progress) {
        return (progress == tileNum - 1) ? lastTileLength : tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t progress) {
        uint32_t length = GetCurLength(progress);
        uint32_t offset = progress * tileLength;

        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm[offset], length);
        DataCopy(yLocal, yGm[offset], length);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress) {
        uint32_t length = GetCurLength(progress);

        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, length);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        uint32_t length = GetCurLength(progress);
        uint32_t offset = progress * tileLength;

        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    // 双缓冲队列
    TQue<QuePosition::VECIN,  2> inQueueX;
    TQue<QuePosition::VECIN,  2> inQueueY;
    TQue<QuePosition::VECOUT, 2> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}