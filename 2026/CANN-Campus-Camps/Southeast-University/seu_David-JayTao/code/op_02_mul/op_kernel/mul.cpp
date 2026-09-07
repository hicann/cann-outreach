// Kernel侧核函数实现

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
        uint32_t blockLength,
        uint32_t tileNum,
        uint32_t tileLength)
    {
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;

        uint32_t blockIdx =
            AscendC::GetBlockIdx();

        // ========================================================
        // 每个 Core 找到属于自己的 Global Memory 区域
        // ========================================================
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

        // ========================================================
        // 给 Queue 分配 Local Memory
        // ========================================================
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount =
            tileNum * BUFFER_NUM;

        for (uint32_t i = 0;
             i < loopCount;
             ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:

    // ============================================================
    // Stage 1：GM -> Local
    // ============================================================
    __aicore__ inline void CopyIn(
        uint32_t progress)
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

    // ============================================================
    // Stage 2：z = x * y
    // ============================================================
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // ========================================================
        // 整道 Mul Kernel 真正独特的地方
        // ========================================================
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // ============================================================
    // Stage 3：Local -> GM
    // ============================================================
    __aicore__ inline void CopyOut(
        uint32_t progress)
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
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};


// ================================================================
// Kernel入口
// ================================================================

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
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tilingData.blockLength,
        tilingData.tileNum,
        tilingData.tileLength);

    op.Process();
}