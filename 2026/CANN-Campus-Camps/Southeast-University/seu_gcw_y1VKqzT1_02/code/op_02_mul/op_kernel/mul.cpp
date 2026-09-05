// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// Double Buffer
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
        // 每个AI Core负责处理的元素数量
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        // 每个Core内部的分块数
        this->tileNum = tileNum;

        // 使用Double Buffer，因此实际每个小Tile的长度还需要除BUFFER_NUM
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 当前Core负责的数据起始位置
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 初始化输入/输出队列
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
        // Double Buffer下：
        // 实际循环次数 = tileNum * BUFFER_NUM
        int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // ========================================================
    // 1. GM -> UB
    // ========================================================

    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从队列申请LocalTensor
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // 将GM数据搬运到UB
        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        // 输入Tensor入队
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // ========================================================
    // 2. z = x * y
    // ========================================================

    __aicore__ inline void Compute(int32_t progress)
    {
        // 输入出队
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 申请输出LocalTensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // ====================================================
        // 核心计算：逐元素乘法
        //
        // z[i] = x[i] * y[i]
        // ====================================================
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 输出入队
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入Tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // ========================================================
    // 3. UB -> GM
    // ========================================================

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 输出出队
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // 搬回GM
        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        // 释放输出Tensor
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // Pipeline
    AscendC::TPipe pipe;

    // 输入队列
    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueY;

    // 输出队列
    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    // Global Memory Tensor
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    // 当前Core处理的总长度
    uint32_t blockLength;

    // Core内部逻辑分块数量
    uint32_t tileNum;

    // 单次计算的数据长度
    uint32_t tileLength;
};


// ============================================================
// Kernel入口
// ============================================================

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // 注册Tiling结构体
    REGISTER_TILING_DEFAULT(MulTilingData);

    // 从Host侧读取Tiling数据
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