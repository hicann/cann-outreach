#include "kernel_operator.h"

// 每个 Queue 使用 1 个 Buffer
constexpr int32_t BUFFER_NUM = 1;

// Queue 深度
constexpr int32_t QUEUE_DEPTH = 1;


template <
    typename dtypeX,
    typename dtypeY,
    typename dtypeZ>
class KernelDivCustomTemplate {
public:

    __aicore__ inline KernelDivCustomTemplate()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength)
    {
        /*
         * Host 侧设置：
         *
         * blockDim = 8
         *
         * 当前测试：
         *
         * totalLength = 8 * 2048
         *             = 16384
         *
         * 所以：
         *
         * blockLength = 16384 / 8
         *             = 2048
         */

        blockLength =
            totalLength / AscendC::GetBlockNum();


        /*
         * 根据当前 Core ID 计算当前 Core
         * 在 Global Memory 中负责的起始位置。
         *
         * Core 0：0
         * Core 1：2048
         * Core 2：4096
         * ...
         */

        uint32_t blockOffset =
            blockLength *
            AscendC::GetBlockIdx();


        // 设置输入 x 的 GM 地址
        xGm.SetGlobalBuffer(
            (__gm__ dtypeX*)x + blockOffset,
            blockLength);


        // 设置输入 y 的 GM 地址
        yGm.SetGlobalBuffer(
            (__gm__ dtypeY*)y + blockOffset,
            blockLength);


        // 设置输出 z 的 GM 地址
        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ*)z + blockOffset,
            blockLength);


        /*
         * 在 UB 中申请输入输出 Buffer。
         *
         * 每个 Core 一次处理 blockLength 个元素。
         */

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            blockLength * sizeof(dtypeX));


        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            blockLength * sizeof(dtypeY));


        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            blockLength * sizeof(dtypeZ));
    }


    __aicore__ inline void Process()
    {
        // GM -> UB
        CopyIn();

        // z = x / y
        Compute();

        // UB -> GM
        CopyOut();
    }


private:

    /*
     * CopyIn
     *
     * Global Memory
     *      ↓
     * Unified Buffer
     */
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.AllocTensor<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.AllocTensor<dtypeY>();


        // x：GM -> UB
        AscendC::DataCopy(
            xLocal,
            xGm,
            blockLength);


        // y：GM -> UB
        AscendC::DataCopy(
            yLocal,
            yGm,
            blockLength);


        // 入队，供 Compute 使用
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    /*
     * Compute
     *
     * z[i] = x[i] / y[i]
     */
    __aicore__ inline void Compute()
    {
        // 从输入 Queue 取出数据
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.DeQue<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.DeQue<dtypeY>();


        // 为输出申请 LocalTensor
        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.AllocTensor<dtypeZ>();


        /*
         * 核心 Vector 计算
         *
         * z = x / y
         */

        AscendC::Div(
            zLocal,
            xLocal,
            yLocal,
            blockLength);


        // 输出数据入队
        outQueueZ.EnQue<dtypeZ>(zLocal);


        // 输入已经计算完成，可以释放
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    /*
     * CopyOut
     *
     * Unified Buffer
     *      ↓
     * Global Memory
     */
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.DeQue<dtypeZ>();


        // UB -> GM
        AscendC::DataCopy(
            zGm,
            zLocal,
            blockLength);


        // 释放输出 LocalTensor
        outQueueZ.FreeTensor(zLocal);
    }


private:

    // 管理 UB Buffer
    AscendC::TPipe pipe;


    // 输入 x Queue
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        QUEUE_DEPTH>
        inQueueX;


    // 输入 y Queue
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        QUEUE_DEPTH>
        inQueueY;


    // 输出 z Queue
    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        QUEUE_DEPTH>
        outQueueZ;


    // Global Memory Tensor
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;


    // 当前 Core 处理的元素数量
    uint32_t blockLength;
};


/*
 * Kernel 入口
 */
extern "C" __global__ __aicore__
void div_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    /*
     * 将 Host 传入的 TilingData 解析为 tiling_data。
     *
     * 其中：
     *
     * tiling_data.size
     *
     * 就是 Host 中设置的：
     *
     * tiling.set_size(dataSize)
     */

    GET_TILING_DATA(
        tiling_data,
        tiling);


    /*
     * DTYPE_X / DTYPE_Y / DTYPE_Z
     *
     * 会由 Ascend C 编译系统根据
     * 算子注册的数据类型自动生成。
     *
     * 对本题：
     *
     * half / half / half
     *
     * 或
     *
     * float / float / float
     */

    KernelDivCustomTemplate<
        DTYPE_X,
        DTYPE_Y,
        DTYPE_Z>
        op;


    op.Init(
        x,
        y,
        z,
        tiling_data.size);


    op.Process();
}