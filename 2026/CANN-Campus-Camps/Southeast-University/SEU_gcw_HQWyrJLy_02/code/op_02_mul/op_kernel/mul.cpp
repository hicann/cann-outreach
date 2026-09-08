// Kernel侧核函数实现

#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"


template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // ============================================================
    // 初始化
    // ============================================================
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        // 本题输入 shape 为 (8, 2048)
        // Host 侧启动 8 个核
        // 因此每个核负责 16384 / 8 = 2048 个元素
        blockLength = length / AscendC::GetBlockNum();

        // 当前核负责的数据起始位置
        uint32_t offset =
            blockLength * AscendC::GetBlockIdx();

        // 绑定输入 x 的 GM
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + offset,
            blockLength);

        // 绑定输入 y 的 GM
        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + offset,
            blockLength);

        // 绑定输出 z 的 GM
        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + offset,
            blockLength);

        // 为输入 x 分配 Local Memory
        pipe.InitBuffer(
            inQueueX,
            1,
            blockLength * sizeof(DT_X));

        // 为输入 y 分配 Local Memory
        pipe.InitBuffer(
            inQueueY,
            1,
            blockLength * sizeof(DT_X));

        // 为输出 z 分配 Local Memory
        pipe.InitBuffer(
            outQueueZ,
            1,
            blockLength * sizeof(DT_X));
    }


    // ============================================================
    // 主处理流程
    // ============================================================
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }


private:

    // ============================================================
    // CopyIn
    // GM -> Local Memory
    // ============================================================
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // x 从 GM 搬入 Local Memory
        AscendC::DataCopy(
            xLocal,
            xGm,
            blockLength);

        // y 从 GM 搬入 Local Memory
        AscendC::DataCopy(
            yLocal,
            yGm,
            blockLength);

        // 放入输入队列
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    // ============================================================
    // Compute
    //
    // z = x * y
    // ============================================================
    __aicore__ inline void Compute()
    {
        // 从队列中取出 x
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        // 从队列中取出 y
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 为 z 分配 LocalTensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // ========================================================
        // 本题核心计算
        //
        // z[i] = x[i] * y[i]
        // ========================================================
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            blockLength);

        // 计算结果进入输出队列
        outQueueZ.EnQue(zLocal);

        // 输入数据已经使用完毕，释放
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    // ============================================================
    // CopyOut
    // Local Memory -> GM
    // ============================================================
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // 将计算结果写回 GM
        AscendC::DataCopy(
            zGm,
            zLocal,
            blockLength);

        // 释放 LocalTensor
        outQueueZ.FreeTensor(zLocal);
    }


private:

    // Pipeline 管理
    AscendC::TPipe pipe;

    // 输入队列 x
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        1> inQueueX;

    // 输入队列 y
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        1> inQueueY;

    // 输出队列 z
    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        1> outQueueZ;

    // Global Memory Tensor
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    // 每个核处理的数据长度
    uint32_t blockLength;
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
    // 注册 Tiling 数据结构
    REGISTER_TILING_DEFAULT(MulTilingData);

    // 从 Host 侧传入的 tiling 地址解析数据
    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    // 创建算子
    KernelMul<DT_X> op;

    // 初始化
    op.Init(
        x,
        y,
        z,
        tiling_data.length);

    // 执行
    op.Process();
}