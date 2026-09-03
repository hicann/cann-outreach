// Kernel侧核函数实现

#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 256;

template <class DT_X>
class KernelMul
{
public:
    __aicore__ inline KernelMul()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        // 当前 Core 编号
        uint32_t blockIdx = GetBlockIdx();

        // 总 Core 数
        uint32_t blockNum = GetBlockNum();

        // 当前 Core 负责的数据量
        blockLength = length / blockNum;

        // 当前 Core 对应的 GM 地址
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

        // Tile 大小
        tileLength = TILE_LENGTH;

        // 当前 Core 处理的 Tile 数量
        tileNum = blockLength / tileLength;

        // 初始化输入输出队列
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
        for (uint32_t i = 0; i < tileNum; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    // =========================================================
    // Global Memory → Local Memory
    // =========================================================
    __aicore__ inline void CopyIn(
        uint32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            xGm[progress * tileLength],
            tileLength);

        DataCopy(
            yLocal,
            yGm[progress * tileLength],
            tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // =========================================================
    // Local Memory 中执行向量乘法
    // =========================================================
    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 核心计算：
        // z = x * y
        Mul(
            zLocal,
            xLocal,
            yLocal,
            tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // =========================================================
    // Local Memory → Global Memory
    // =========================================================
    __aicore__ inline void CopyOut(
        uint32_t progress)
    {
        LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        DataCopy(
            zGm[progress * tileLength],
            zLocal,
            tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;

    TQue<TPosition::VECIN, BUFFER_NUM>
        inQueueX;

    TQue<TPosition::VECIN, BUFFER_NUM>
        inQueueY;

    TQue<TPosition::VECOUT, BUFFER_NUM>
        outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

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

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.length);

    op.Process();
}