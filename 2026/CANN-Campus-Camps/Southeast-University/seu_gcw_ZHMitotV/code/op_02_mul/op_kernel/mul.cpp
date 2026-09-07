// Kernel侧核函数实现

#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"


// ================================================================
// Double Buffer
// ================================================================

constexpr uint32_t BUFFER_NUM = 2;


// ================================================================
// KernelMul
// ================================================================

template <class DT_X>
class KernelMul {

public:

    // ------------------------------------------------------------
    // 构造函数
    // ------------------------------------------------------------

    __aicore__ inline KernelMul()
    {
    }


    // ------------------------------------------------------------
    // Init
    // ------------------------------------------------------------

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t tileNum)
    {
        // ========================================================
        // 1. 计算当前Block负责的数据量
        // ========================================================

        this->blockLength =
            length / AscendC::GetBlockNum();

        // 保存tile数量
        this->tileNum = tileNum;

        // ========================================================
        // 2. 计算每个tile的数据量
        // ========================================================

        /*
         * 一个Block：
         *
         * blockLength
         *
         * 分成：
         *
         * tileNum
         *
         * 再考虑双缓冲：
         *
         * BUFFER_NUM = 2
         */

        this->tileLength =
            this->blockLength
            / this->tileNum
            / BUFFER_NUM;


        // ========================================================
        // 3. 设置Global Memory
        // ========================================================

        /*
         * 每一个AIV核只处理自己对应的Block。
         *
         * Block 0：
         *
         *     x[0 ... blockLength-1]
         *
         * Block 1：
         *
         *     x[blockLength ... 2*blockLength-1]
         *
         * ...
         */

        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x
                + this->blockLength
                * AscendC::GetBlockIdx(),
            this->blockLength
        );


        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y
                + this->blockLength
                * AscendC::GetBlockIdx(),
            this->blockLength
        );


        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z
                + this->blockLength
                * AscendC::GetBlockIdx(),
            this->blockLength
        );


        // ========================================================
        // 4. 初始化Pipe和Queue
        // ========================================================

        /*
         * x：
         *
         * GM -> Local
         */

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength
                * sizeof(DT_X)
        );


        /*
         * y：
         *
         * GM -> Local
         */

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength
                * sizeof(DT_X)
        );


        /*
         * z：
         *
         * Local -> GM
         */

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength
                * sizeof(DT_X)
        );
    }


    // ------------------------------------------------------------
    // Process
    // ------------------------------------------------------------

    __aicore__ inline void Process()
    {
        /*
         * 每一个Block：
         *
         * tileNum个tile
         *
         * 每个tile考虑BUFFER_NUM=2
         *
         * 所以总循环次数：
         *
         * tileNum * BUFFER_NUM
         */

        int32_t loopCount =
            this->tileNum * BUFFER_NUM;


        for (int32_t i = 0;
             i < loopCount;
             ++i)
        {
            // GM -> Local
            CopyIn(i);

            // Local计算
            Compute(i);

            // Local -> GM
            CopyOut(i);
        }
    }


private:

    // ============================================================
    // CopyIn
    // ============================================================

    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        // --------------------------------------------------------
        // 为x申请LocalTensor
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();


        // --------------------------------------------------------
        // 为y申请LocalTensor
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();


        // --------------------------------------------------------
        // x: GM -> Local
        // --------------------------------------------------------

        AscendC::DataCopy(
            xLocal,
            xGm[
                progress
                * this->tileLength
            ],
            this->tileLength
        );


        // --------------------------------------------------------
        // y: GM -> Local
        // --------------------------------------------------------

        AscendC::DataCopy(
            yLocal,
            yGm[
                progress
                * this->tileLength
            ],
            this->tileLength
        );


        // --------------------------------------------------------
        // 放入输入队列
        // --------------------------------------------------------

        inQueueX.EnQue<DT_X>(
            xLocal
        );

        inQueueY.EnQue<DT_X>(
            yLocal
        );
    }


    // ============================================================
    // Compute
    // ============================================================

    __aicore__ inline void Compute(
        int32_t progress)
    {
        // --------------------------------------------------------
        // 从输入队列取出x
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();


        // --------------------------------------------------------
        // 从输入队列取出y
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();


        // --------------------------------------------------------
        // 为输出申请LocalTensor
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();


        // --------------------------------------------------------
        // 核心计算
        //
        // zLocal = xLocal * yLocal
        // --------------------------------------------------------

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength
        );


        // --------------------------------------------------------
        // 将计算结果放入输出队列
        // --------------------------------------------------------

        outQueueZ.EnQue<DT_X>(
            zLocal
        );


        // --------------------------------------------------------
        // 释放输入LocalTensor
        // --------------------------------------------------------

        inQueueX.FreeTensor(
            xLocal
        );

        inQueueY.FreeTensor(
            yLocal
        );
    }


    // ============================================================
    // CopyOut
    // ============================================================

    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        // --------------------------------------------------------
        // 从输出队列取出结果
        // --------------------------------------------------------

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();


        // --------------------------------------------------------
        // Local -> GM
        // --------------------------------------------------------

        AscendC::DataCopy(
            zGm[
                progress
                * this->tileLength
            ],
            zLocal,
            this->tileLength
        );


        // --------------------------------------------------------
        // 释放输出LocalTensor
        // --------------------------------------------------------

        outQueueZ.FreeTensor(
            zLocal
        );
    }


private:

    // ============================================================
    // Pipe
    // ============================================================

    AscendC::TPipe pipe;


    // ============================================================
    // 输入队列
    // ============================================================

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM
    > inQueueX;


    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM
    > inQueueY;


    // ============================================================
    // 输出队列
    // ============================================================

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM
    > outQueueZ;


    // ============================================================
    // Global Memory
    // ============================================================

    AscendC::GlobalTensor<DT_X> xGm;

    AscendC::GlobalTensor<DT_X> yGm;

    AscendC::GlobalTensor<DT_X> zGm;


    // ============================================================
    // Tiling参数
    // ============================================================

    uint32_t blockLength = 0;

    uint32_t tileNum = 0;

    uint32_t tileLength = 0;
};


// ================================================================
// Kernel入口
// ================================================================

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // ------------------------------------------------------------
    // 注册Tiling结构
    // ------------------------------------------------------------

    REGISTER_TILING_DEFAULT(MulTilingData);


    // ------------------------------------------------------------
    // 获取Tiling数据
    // ------------------------------------------------------------

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling
    );


    // ------------------------------------------------------------
    // 创建Kernel对象
    // ------------------------------------------------------------

    KernelMul<DT_X> op;


    // ------------------------------------------------------------
    // 初始化
    // ------------------------------------------------------------

    op.Init(
        x,
        y,
        z,
        tiling_data.length,
        tiling_data.tileNum
    );


    // ------------------------------------------------------------
    // 执行计算
    // ------------------------------------------------------------

    op.Process();
}