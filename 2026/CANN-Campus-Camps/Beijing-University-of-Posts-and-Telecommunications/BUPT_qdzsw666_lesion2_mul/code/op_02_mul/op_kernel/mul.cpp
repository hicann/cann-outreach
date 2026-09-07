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
        // 每个核负责的数据长度
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        // 每个核划分的Tile数量
        this->tileNum = tileNum;

        // 循环次数为tileNum × BUFFER_NUM
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 当前核对应的GM起始偏移
        uint32_t blockOffset =
            this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + blockOffset,
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + blockOffset,
            this->blockLength);

        // 为输入和输出队列分配UB空间
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
        // 实际数据块数为tileNum × BUFFER_NUM
        int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从输入队列申请LocalTensor
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        uint32_t offset =
            progress * this->tileLength;

        // 将x和y从GM搬入UB
        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[offset],
            this->tileLength);

        // 输入数据入队
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 为输出结果申请LocalTensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 执行逐元素乘法：z = x * y
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 计算结果放入输出队列
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入LocalTensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列获取计算结果
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t offset =
            progress * this->tileLength;

        // 将结果从UB搬回GM
        AscendC::DataCopy(
            zGm[offset],
            zLocal,
            this->tileLength);

        // 释放输出LocalTensor
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // UB内存管理对象
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

    uint32_t blockLength; // 每个核负责的元素数
    uint32_t tileNum;     // 每个核内的Tile数量
    uint32_t tileLength;  // 每次循环处理的元素数
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