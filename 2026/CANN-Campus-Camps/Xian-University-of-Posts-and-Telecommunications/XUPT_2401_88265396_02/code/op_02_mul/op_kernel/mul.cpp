// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"


// ============================================================
// KernelMul
// ============================================================

template <
    class DT_X,
    class DT_Y,
    class DT_Z>
class KernelMul {
public:

    __aicore__ inline KernelMul()
    {
    }


    // ========================================================
    // Init
    // ========================================================

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        totalLength =
            length;


        // ----------------------------------------------------
        // 整个Tensor绑定到GlobalTensor。
        //
        // 不再按8核提前切分。
        // ----------------------------------------------------

        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x,
            totalLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_Y *)y,
            totalLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_Z *)z,
            totalLength);


        // ----------------------------------------------------
        // 输入x
        // ----------------------------------------------------

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            TILE_LENGTH *
                sizeof(DT_X));


        // ----------------------------------------------------
        // 输入y
        // ----------------------------------------------------

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            TILE_LENGTH *
                sizeof(DT_Y));


        // ----------------------------------------------------
        // 输出z
        // ----------------------------------------------------

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            TILE_LENGTH *
                sizeof(DT_Z));
    }


    // ========================================================
    // Process
    // ========================================================

    __aicore__ inline void Process()
    {
        /*
         * 不再假设：
         *
         * length == 16384
         *
         * 也不再假设：
         *
         * length % 8 == 0
         *
         * 从0开始一直处理到totalLength结束。
         */

        uint32_t offset = 0;


        while (offset < totalLength) {

            // ------------------------------------------------
            // 当前Tile真实元素数量
            // ------------------------------------------------

            uint32_t currentLength =
                totalLength - offset;


            if (currentLength >
                TILE_LENGTH) {

                currentLength =
                    TILE_LENGTH;
            }


            // ------------------------------------------------
            // GM -> UB
            // ------------------------------------------------

            CopyIn(
                offset,
                currentLength);


            // ------------------------------------------------
            // z = x * y
            // ------------------------------------------------

            Compute(
                currentLength);


            // ------------------------------------------------
            // UB -> GM
            // ------------------------------------------------

            CopyOut(
                offset,
                currentLength);


            // ------------------------------------------------
            // 下一个Tile
            // ------------------------------------------------

            offset +=
                currentLength;
        }
    }


private:

    // ========================================================
    // CopyIn
    // ========================================================

    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t length)
    {
        // ----------------------------------------------------
        // LocalTensor x
        // ----------------------------------------------------

        AscendC::LocalTensor<DT_X>
            xLocal =
                inQueueX
                    .AllocTensor<
                        DT_X>();


        // ----------------------------------------------------
        // LocalTensor y
        // ----------------------------------------------------

        AscendC::LocalTensor<DT_Y>
            yLocal =
                inQueueY
                    .AllocTensor<
                        DT_Y>();


        // ====================================================
        // x 搬运参数
        //
        // blockLen单位是Byte。
        //
        // DataCopyPad支持非32Byte对齐的数据搬运。
        // ====================================================

        AscendC::DataCopyExtParams
            copyXParams{
                1,

                static_cast<uint32_t>(
                    length *
                    sizeof(DT_X)),

                0,
                0,
                0
            };


        AscendC::DataCopyPadExtParams<
            DT_X>
            padXParams{
                false,
                0,
                0,
                0
            };


        // ====================================================
        // y 搬运参数
        // ====================================================

        AscendC::DataCopyExtParams
            copyYParams{
                1,

                static_cast<uint32_t>(
                    length *
                    sizeof(DT_Y)),

                0,
                0,
                0
            };


        AscendC::DataCopyPadExtParams<
            DT_Y>
            padYParams{
                false,
                0,
                0,
                0
            };


        // ====================================================
        // x:
        //
        // GM -> VECIN
        // ====================================================

        AscendC::DataCopyPad(
            xLocal,
            xGm[offset],
            copyXParams,
            padXParams);


        // ====================================================
        // y:
        //
        // GM -> VECIN
        // ====================================================

        AscendC::DataCopyPad(
            yLocal,
            yGm[offset],
            copyYParams,
            padYParams);


        // ----------------------------------------------------
        // 入队
        // ----------------------------------------------------

        inQueueX.EnQue(
            xLocal);

        inQueueY.EnQue(
            yLocal);
    }


    // ========================================================
    // Compute
    // ========================================================

    __aicore__ inline void Compute(
        uint32_t length)
    {
        // ----------------------------------------------------
        // 获取x
        // ----------------------------------------------------

        AscendC::LocalTensor<DT_X>
            xLocal =
                inQueueX
                    .DeQue<
                        DT_X>();


        // ----------------------------------------------------
        // 获取y
        // ----------------------------------------------------

        AscendC::LocalTensor<DT_Y>
            yLocal =
                inQueueY
                    .DeQue<
                        DT_Y>();


        // ----------------------------------------------------
        // 获取输出z
        // ----------------------------------------------------

        AscendC::LocalTensor<DT_Z>
            zLocal =
                outQueueZ
                    .AllocTensor<
                        DT_Z>();


        // ====================================================
        // 核心计算
        //
        // z[i] = x[i] * y[i]
        //
        // 只计算当前Tile真实存在的length个元素。
        // ====================================================

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            length);


        // ----------------------------------------------------
        // 输出入队
        // ----------------------------------------------------

        outQueueZ.EnQue(
            zLocal);


        // ----------------------------------------------------
        // 释放输入
        // ----------------------------------------------------

        inQueueX.FreeTensor(
            xLocal);

        inQueueY.FreeTensor(
            yLocal);
    }


    // ========================================================
    // CopyOut
    // ========================================================

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t length)
    {
        AscendC::LocalTensor<DT_Z>
            zLocal =
                outQueueZ
                    .DeQue<
                        DT_Z>();


        // ====================================================
        // 只搬出当前Tile真实有效的数据。
        //
        // DataCopyPad允许非32Byte对齐。
        // ====================================================

        AscendC::DataCopyExtParams
            copyParams{
                1,

                static_cast<uint32_t>(
                    length *
                    sizeof(DT_Z)),

                0,
                0,
                0
            };


        AscendC::DataCopyPad(
            zGm[offset],
            zLocal,
            copyParams);


        // ----------------------------------------------------
        // 释放输出Tensor
        // ----------------------------------------------------

        outQueueZ.FreeTensor(
            zLocal);
    }


private:

    // ========================================================
    // BUFFER_NUM
    //
    // 当前正确性优先，使用单Buffer。
    // ========================================================

    static constexpr uint32_t
        BUFFER_NUM = 1;


    // ========================================================
    // 每个Tile最多2048个元素。
    //
    // FP32：
    //
    // x  = 8192 B
    // y  = 8192 B
    // z  = 8192 B
    //
    // 总共约24KB。
    //
    // FP16则更小。
    // ========================================================

    static constexpr uint32_t
        TILE_LENGTH = 2048;


    // ========================================================
    // Pipeline
    // ========================================================

    AscendC::TPipe pipe;


    // ========================================================
    // 输入x
    // ========================================================

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM>
        inQueueX;


    // ========================================================
    // 输入y
    // ========================================================

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM>
        inQueueY;


    // ========================================================
    // 输出z
    // ========================================================

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM>
        outQueueZ;


    // ========================================================
    // GM Tensor
    // ========================================================

    AscendC::GlobalTensor<DT_X>
        xGm;

    AscendC::GlobalTensor<DT_Y>
        yGm;

    AscendC::GlobalTensor<DT_Z>
        zGm;


    // ========================================================
    // 总元素数量
    // ========================================================

    uint32_t totalLength;
};



// ============================================================
// Kernel入口
// ============================================================

template <
    typename DT_X,
    typename DT_Y,
    typename DT_Z>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // --------------------------------------------------------
    // 注册TilingData
    // --------------------------------------------------------

    REGISTER_TILING_DEFAULT(
        MulTilingData);


    // --------------------------------------------------------
    // 读取Host下发的Tiling数据
    // --------------------------------------------------------

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);


    // --------------------------------------------------------
    // 创建Kernel
    // --------------------------------------------------------

    KernelMul<
        DT_X,
        DT_Y,
        DT_Z>
        op;


    // --------------------------------------------------------
    // 初始化
    // --------------------------------------------------------

    op.Init(
        x,
        y,
        z,
        tiling_data.length);


    // --------------------------------------------------------
    // 执行
    // --------------------------------------------------------

    op.Process();
}