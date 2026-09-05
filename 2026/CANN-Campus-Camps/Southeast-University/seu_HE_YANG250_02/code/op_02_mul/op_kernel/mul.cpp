// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;


template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul()
    {
    }

    // ================================================================
    // Init
    // ================================================================
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

        /*
         * 每个Core负责一段连续数据。
         *
         * 题目标准场景：
         *
         * Core 0 -> [0, 2048)
         * Core 1 -> [2048, 4096)
         * ...
         * Core 7 -> [14336, 16384)
         */
        uint32_t blockOffset =
            AscendC::GetBlockIdx() *
            this->blockLength;

        // ============================================================
        // 设置GM地址
        // ============================================================
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + blockOffset,
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + blockOffset,
            this->blockLength);

        // ============================================================
        // 初始化UB Queue
        //
        // BUFFER_NUM = 2
        // 开启Double Buffer
        // ============================================================
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


    // ================================================================
    // Process
    // ================================================================
    __aicore__ inline void Process()
    {
        /*
         * 因为开启了Double Buffer：
         *
         * 实际循环次数：
         *
         * tileNum * BUFFER_NUM
         *
         * 标准场景：
         *
         * 8 * 2 = 16
         */
        int32_t loopCount =
            static_cast<int32_t>(
                this->tileNum *
                BUFFER_NUM);

        for (int32_t i = 0;
             i < loopCount;
             ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }


private:

    // ================================================================
    // CopyIn
    //
    // GM -> UB
    // ================================================================
    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        uint32_t offset =
            static_cast<uint32_t>(progress) *
            this->tileLength;

        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[offset],
            this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    // ================================================================
    // Compute
    //
    // z = x * y
    // ================================================================
    __aicore__ inline void Compute(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        /*
         * Ascend C Vector Mul
         *
         * zLocal[i] =
         *     xLocal[i] * yLocal[i]
         */
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    // ================================================================
    // CopyOut
    //
    // UB -> GM
    // ================================================================
    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t offset =
            static_cast<uint32_t>(progress) *
            this->tileLength;

        AscendC::DataCopy(
            zGm[offset],
            zLocal,
            this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }


private:

    // ================================================================
    // Pipeline
    // ================================================================
    AscendC::TPipe pipe;


    // ================================================================
    // Queue
    // ================================================================
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> outQueueZ;


    // ================================================================
    // Global Memory Tensor
    // ================================================================
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;


    // ================================================================
    // Tiling参数
    // ================================================================
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


// ====================================================================
// Kernel入口
//
// DT_X由tiling_key_mul.h控制：
//
// float16 -> half
// float32 -> float
//
// ====================================================================

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
