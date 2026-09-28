/**
 * @file add_custom.cpp
 *
 * @brief add_custom 算子核函数（device 侧，Ascend C）
 *
 * 功能：z = x + y，逐元素加法
 * 输入：x, y（2D [N2, N1]，float16，ND）
 * 输出：z（shape 同 x）
 *
 * 实现说明：ND 格式下张量在 GM 中按行主序连续存放，
 * 因此将 2D 张量展平为一维长度 totalLength = N2 * N1，
 * 按 block 切分后走 "搬入(CopyIn) -> 计算(Compute) -> 搬出(CopyOut)"
 * 三级流水，双缓冲（BUFFER_NUM=2）实现搬运与计算重叠。
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    /**
     * 初始化：计算切分参数、绑定 GM 地址、为 UB 队列分配内存
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // blockLength: 每个核（AI Core）上负责计算的元素总数
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // tileLength: 每个核上每个分块的元素数（除以 BUFFER_NUM 做双缓冲）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 按核切分 GM 地址：本核只处理自己负责的那一段
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 通过 Pipe 内存管理对象为输入输出 Queue 分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }

    /**
     * 核心处理：循环执行 "搬入 -> 计算 -> 搬出" 三级流水
     */
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /**
     * 搬入：GM -> UB（VECIN 队列）
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    /**
     * 计算：调用 AscendC::Add 接口做逐元素加法，结果放入 VECOUT 队列
     */
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        // z = x + y
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    /**
     * 搬出：UB（VECOUT 队列）-> GM
     */
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // Pipe 内存管理对象
    AscendC::TPipe pipe;
    // 输入数据 Queue（VECIN 位置），双缓冲
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    // 输出数据 Queue（VECOUT 位置），双缓冲
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    // 输入输出 GM 地址管理对象
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    // 每个核上总计算数据大小（元素数）
    uint32_t blockLength;
    // 每个核上总计算数据分块个数
    uint32_t tileNum;
    // 每个分块大小（元素数）
    uint32_t tileLength;
};

/**
 * 算子核函数入口：由调度框架调用
 * @param x        输入张量 x 的 GM 地址
 * @param y        输入张量 y 的 GM 地址
 * @param z        输出张量 z 的 GM 地址
 * @param workspace 工作空间地址（本算子未使用）
 * @param tiling   Host 侧传入的 Tiling 参数地址
 */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 获取 Host 侧传入的 Tiling 参数
    GET_TILING_DATA(tiling_data, tiling);

    KernelAddCustom op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
