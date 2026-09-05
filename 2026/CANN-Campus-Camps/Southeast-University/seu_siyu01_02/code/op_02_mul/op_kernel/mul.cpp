// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 每个输入/输出队列分配两个 buffer，用于双缓冲。
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
        // 每个核处理 totalLength / 核数 个元素。
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每个 tile 的大小需要除以双缓冲数量。
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 将输入输出全局内存按核划分为连续的数据区间。
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);
        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);
        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 初始化输入和输出队列，为每个 buffer 分配 tileLength 个元素。
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 依次完成数据搬入、矢量乘法和结果搬出。
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从 Global Memory 搬入当前 tile，并放入输入队列。
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

    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据，执行逐元素乘法 z = x * y。
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 结果入队，同时释放已经使用完的输入 buffer。
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 将计算结果从输出队列搬回 Global Memory。
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
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
    // 读取 host 侧生成的 tiling 数据并启动乘法计算。
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
