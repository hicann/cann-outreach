// Kernel侧核函数实现

#include "kernel_operator.h"
#include "add_tiling.h"
#include "tiling_key_add.h"

constexpr int32_t BUFFER_NUM = 2;


template <class DT_X>
class KernelAdd {

public:
    __aicore__ inline KernelAdd()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t tile_num)
    {
        // 每个 Core 负责的数据量
        this->blockLength =
            length / AscendC::GetBlockNum();

        // 每个 Core 划分的 Tile 数
        this->tileNum = tile_num;

        // 每个 Buffer 实际处理的数据量
        //
        // length = 16384
        // blockLength = 16384 / 8 = 2048
        // tileNum = 8
        // BUFFER_NUM = 2
        //
        // tileLength = 2048 / 8 / 2 = 128
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        // 当前 Core 对应的 GM 起始地址
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

        // 初始化输入 Queue
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        // 初始化输出 Queue
        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 每个 Core：
        //
        // TILE_NUM * BUFFER_NUM
        // = 8 * 2
        // = 16 次
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
        // 从输入 Queue 获取 Local Tensor
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // GM -> UB
        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        // 放入 Queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 从 Queue 中取出输入
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 分配输出 Tensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // z = x + y
        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 输出进入 Queue
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出 Queue 取出结果
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // UB -> GM
        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        // 释放输出 Tensor
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

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


template <typename DT_X>
__global__ __aicore__ void add(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // 注册默认 Tiling 结构体
    REGISTER_TILING_DEFAULT(AddTilingData);

    // 获取 Tiling 数据
    GET_TILING_DATA_WITH_STRUCT(
        AddTilingData,
        tiling_data,
        tiling);

    // 创建 Kernel
    KernelAdd<DT_X> op;

    // 初始化
    op.Init(
        x,
        y,
        z,
        tiling_data.length,
        tiling_data.tile_num);

    // 执行
    op.Process();
}