// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd()
    {
    }

    /*
     * ============================================================
     * Init
     * ============================================================
     */
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t blockLength,
        uint32_t tileLength)
    {
        blockLength_ =
            blockLength;

        tileLength_ =
            tileLength;

        /*
         * 当前Core在整个Tensor中的起始元素位置。
         *
         * 例如：
         *
         * blockLength = 1024
         *
         * Core 0 -> offset 0
         * Core 1 -> offset 1024
         * Core 2 -> offset 2048
         * ...
         */
        uint32_t gmOffset =
            GetBlockIdx() *
            blockLength_;

        /*
         * 分别绑定当前Core负责的GM区域。
         */
        xGm_.SetGlobalBuffer(
            (__gm__ DT_X *)x +
                gmOffset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            (__gm__ DT_X *)y +
                gmOffset,
            blockLength_);

        zGm_.SetGlobalBuffer(
            (__gm__ DT_X *)z +
                gmOffset,
            blockLength_);

        /*
         * ========================================================
         * UB Queue
         * ========================================================
         *
         * X、Y、Z均使用Double Buffer。
         *
         * X:
         *     VECIN × 2
         *
         * Y:
         *     VECIN × 2
         *
         * Z:
         *     VECOUT × 2
         */
        pipe_.InitBuffer(
            xQueue_,
            BUFFER_NUM,
            tileLength_ *
                sizeof(DT_X));

        pipe_.InitBuffer(
            yQueue_,
            BUFFER_NUM,
            tileLength_ *
                sizeof(DT_X));

        pipe_.InitBuffer(
            zQueue_,
            BUFFER_NUM,
            tileLength_ *
                sizeof(DT_X));
    }

    /*
     * ============================================================
     * Process
     * ============================================================
     */
    __aicore__ inline void Process()
    {
        uint32_t progress = 0;

        while (progress < blockLength_) {
            uint32_t currentLength =
                blockLength_ - progress;

            if (currentLength >
                tileLength_) {
                currentLength =
                    tileLength_;
            }

            CopyIn(
                progress,
                currentLength);

            Compute(
                currentLength);

            CopyOut(
                progress,
                currentLength);

            progress +=
                currentLength;
        }
    }

private:

    /*
     * ============================================================
     * CopyIn
     *
     * GM -> UB
     * ============================================================
     */
    __aicore__ inline void CopyIn(
        uint32_t progress,
        uint32_t currentLength)
    {
        LocalTensor<DT_X> xLocal =
            xQueue_.AllocTensor<DT_X>();

        LocalTensor<DT_X> yLocal =
            yQueue_.AllocTensor<DT_X>();

        /*
         * x:
         *
         * GM -> VECIN
         */
        DataCopy(
            xLocal,
            xGm_[progress],
            currentLength);

        /*
         * y:
         *
         * GM -> VECIN
         */
        DataCopy(
            yLocal,
            yGm_[progress],
            currentLength);

        /*
         * 入队，交给Compute阶段。
         */
        xQueue_.EnQue(
            xLocal);

        yQueue_.EnQue(
            yLocal);
    }


    /*
     * ============================================================
     * Compute
     *
     * z = x + y
     * ============================================================
     */
    __aicore__ inline void Compute(
        uint32_t currentLength)
    {
        /*
         * 获取当前输入Tile。
         */
        LocalTensor<DT_X> xLocal =
            xQueue_.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            yQueue_.DeQue<DT_X>();

        /*
         * 为输出分配UB。
         */
        LocalTensor<DT_X> zLocal =
            zQueue_.AllocTensor<DT_X>();

        /*
         * 向量加法：
         *
         * z[i] = x[i] + y[i]
         */
        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            static_cast<int32_t>(
                currentLength));

        /*
         * 输出入队。
         */
        zQueue_.EnQue(
            zLocal);

        /*
         * 当前输入已经使用完成，
         * 可以释放。
         */
        xQueue_.FreeTensor(
            xLocal);

        yQueue_.FreeTensor(
            yLocal);
    }


    /*
     * ============================================================
     * CopyOut
     *
     * UB -> GM
     * ============================================================
     */
    __aicore__ inline void CopyOut(
        uint32_t progress,
        uint32_t currentLength)
    {
        LocalTensor<DT_X> zLocal =
            zQueue_.DeQue<DT_X>();

        DataCopy(
            zGm_[progress],
            zLocal,
            currentLength);

        zQueue_.FreeTensor(
            zLocal);
    }

private:

    /*
     * ============================================================
     * Pipeline
     * ============================================================
     */
    TPipe pipe_;

    /*
     * 输入Queue：
     *
     * x
     * y
     */
    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        xQueue_;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        yQueue_;

    /*
     * 输出Queue：
     *
     * z
     */
    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        zQueue_;

    /*
     * GM Tensor。
     */
    GlobalTensor<DT_X>
        xGm_;

    GlobalTensor<DT_X>
        yGm_;

    GlobalTensor<DT_X>
        zGm_;

    /*
     * 当前Core处理的数据量。
     */
    uint32_t blockLength_ = 0;

    /*
     * 单次UB Tile的数据量。
     */
    uint32_t tileLength_ = 0;
};


/*
 * ================================================================
 * Kernel入口
 * ================================================================
 */
template <typename DT_X>
__global__ __aicore__ void add(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        AddTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        AddTilingData,
        tilingData,
        tiling);

    KernelAdd<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tilingData.blockLength,
        tilingData.tileLength);

    op.Process();
}