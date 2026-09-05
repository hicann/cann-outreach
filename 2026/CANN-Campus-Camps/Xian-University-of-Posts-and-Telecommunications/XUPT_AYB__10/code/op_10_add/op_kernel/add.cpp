// Kernel侧核函数实现

#include "kernel_operator.h"
#include "add_tiling.h"
#include "tiling_key_add.h"

constexpr int32_t BUFFER_NUM = 2;  // 每个Queue的Buffer数量


template <class DT_X>
class KernelAdd {

public:
    __aicore__ inline KernelAdd()
    {
    }

    // 初始化
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 获取当前AI Core需要处理的数据长度
        this->blockLength =
            totalLength / AscendC::GetBlockNum();

        // 保存分块数量
        this->tileNum = tileNum;

        // 每次实际处理的数据量
        //
        // totalLength = 16384
        // blockNum    = 8
        // blockLength = 2048
        // tileNum     = 8
        // BUFFER_NUM  = 2
        //
        // tileLength = 2048 / 8 / 2 = 128
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 根据当前Core编号确定GM数据起始位置
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

        // 初始化输入/输出Queue
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

    // 执行计算
    __aicore__ inline void Process()
    {
        // 每个Core总共需要处理blockLength个元素
        //
        // 每次处理tileLength个元素
        //
        // loopCount = blockLength / tileLength
        //           = tileNum * BUFFER_NUM
        //
        // 当前题目：
        // loopCount = 8 * 2 = 16
        int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:

    // GM -> Local Memory
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从输入Queue申请LocalTensor
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // 将x数据搬运到Local Memory
        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        // 将y数据搬运到Local Memory
        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        // 数据进入Queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // Local Memory执行向量加法
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入Queue获取数据
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        // 申请输出LocalTensor
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 执行向量加法
        // z = x + y
        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        // 将计算结果放入输出Queue
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入Tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // Local Memory -> GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出Queue获取计算结果
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // 将结果写回GM
        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        // 释放输出Tensor
        outQueueZ.FreeTensor(zLocal);
    }

private:

    // Pipe
    AscendC::TPipe pipe;

    // 输入Queue
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>
        inQueueX;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>
        inQueueY;

    // 输出Queue
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM>
        outQueueZ;

    // Global Memory Tensor
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    // 当前Core处理的数据总长度
    uint32_t blockLength;

    // 当前Core的分块数量
    uint32_t tileNum;

    // 每次处理的数据长度
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
    // 注册默认Tiling结构
    REGISTER_TILING_DEFAULT(AddTilingData);

    // 获取Tiling数据
    GET_TILING_DATA_WITH_STRUCT(
        AddTilingData,
        tiling_data,
        tiling);

    // 创建Kernel对象
    KernelAdd<DT_X> op;

    // 初始化Kernel
    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.tileNum);

    // 执行计算
    op.Process();
}
