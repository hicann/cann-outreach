#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"


template <class DT_INPUT_X>
class KernelGelu {

public:

    // =========================================================
    // Constructor
    // =========================================================
    __aicore__ inline KernelGelu()
    {
    }


    // =========================================================
    // Init
    // =========================================================
    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length,
        uint32_t tileLength)
    {
        totalLength = length;
        this->tileLength = tileLength;

        // =====================================================
        // Global Memory
        // =====================================================
        inputGlobal.SetGlobalBuffer(
            reinterpret_cast<
                __gm__ DT_INPUT_X *
            >(input_x),
            length);

        outputGlobal.SetGlobalBuffer(
            reinterpret_cast<
                __gm__ DT_INPUT_X *
            >(output),
            length);

        // =====================================================
        // Double Buffer
        // =====================================================
        pipe.InitBuffer(
            inputQueue,
            2,
            tileLength *
            sizeof(DT_INPUT_X));

        pipe.InitBuffer(
            outputQueue,
            2,
            tileLength *
            sizeof(DT_INPUT_X));

        // =====================================================
        // Erf 中间数据
        // =====================================================
        pipe.InitBuffer(
            factorBuffer,
            tileLength *
            sizeof(DT_INPUT_X));
    }


    // =========================================================
    // Exact GELU
    //
    // GELU(x)
    // = 0.5 * x *
    //   (1 + erf(x / sqrt(2)))
    //
    // 保持已经通过验证的精确计算方式。
    // =========================================================
    __aicore__ inline void ComputeGelu(
        AscendC::LocalTensor<DT_INPUT_X> &srcLocal,
        AscendC::LocalTensor<DT_INPUT_X> &dstLocal,
        uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X>
            factorLocal =
                factorBuffer.Get<DT_INPUT_X>();

        // =====================================================
        // 1. x / sqrt(2)
        // =====================================================
        DT_INPUT_X invSqrt2 =
            static_cast<DT_INPUT_X>(
                0.7071067811865475f);

        AscendC::Muls(
            dstLocal,
            srcLocal,
            invSqrt2,
            count);

        // =====================================================
        // 2. erf(x / sqrt(2))
        // =====================================================
        AscendC::Erf(
            factorLocal,
            dstLocal,
            count);

        // =====================================================
        // 3. 1 + erf(...)
        // =====================================================
        DT_INPUT_X one =
            static_cast<DT_INPUT_X>(
                1.0f);

        AscendC::Adds(
            dstLocal,
            factorLocal,
            one,
            count);

        // =====================================================
        // 4. 0.5 * (...)
        // =====================================================
        DT_INPUT_X half =
            static_cast<DT_INPUT_X>(
                0.5f);

        AscendC::Muls(
            factorLocal,
            dstLocal,
            half,
            count);

        // =====================================================
        // 5. x * factor
        // =====================================================
        AscendC::Mul(
            dstLocal,
            srcLocal,
            factorLocal,
            count);
    }


    // =========================================================
    // CopyIn
    // =========================================================
    __aicore__ inline void CopyIn(
        uint32_t progress,
        uint32_t start)
    {
        uint32_t offset =
            start +
            progress *
            tileLength;

        uint32_t remain =
            endLength -
            progress *
            tileLength;

        uint32_t currentLength =
            remain > tileLength
                ? tileLength
                : remain;

        AscendC::LocalTensor<DT_INPUT_X>
            srcLocal =
                inputQueue
                .AllocTensor<DT_INPUT_X>();

        AscendC::DataCopy(
            srcLocal,
            inputGlobal[offset],
            currentLength);

        inputQueue.EnQue(
            srcLocal);
    }


    // =========================================================
    // Compute
    // =========================================================
    __aicore__ inline void Compute(
        uint32_t progress)
    {
        uint32_t remain =
            endLength -
            progress *
            tileLength;

        uint32_t currentLength =
            remain > tileLength
                ? tileLength
                : remain;

        AscendC::LocalTensor<DT_INPUT_X>
            srcLocal =
                inputQueue
                .DeQue<DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X>
            dstLocal =
                outputQueue
                .AllocTensor<DT_INPUT_X>();

        ComputeGelu(
            srcLocal,
            dstLocal,
            currentLength);

        outputQueue.EnQue(
            dstLocal);

        inputQueue.FreeTensor(
            srcLocal);
    }


    // =========================================================
    // CopyOut
    // =========================================================
    __aicore__ inline void CopyOut(
        uint32_t progress,
        uint32_t start)
    {
        uint32_t offset =
            start +
            progress *
            tileLength;

        uint32_t remain =
            endLength -
            progress *
            tileLength;

        uint32_t currentLength =
            remain > tileLength
                ? tileLength
                : remain;

        AscendC::LocalTensor<DT_INPUT_X>
            dstLocal =
                outputQueue
                .DeQue<DT_INPUT_X>();

        AscendC::DataCopy(
            outputGlobal[offset],
            dstLocal,
            currentLength);

        outputQueue.FreeTensor(
            dstLocal);
    }


    // =========================================================
    // Process
    // =========================================================
    __aicore__ inline void Process()
    {
        uint32_t blockIdx =
            AscendC::GetBlockIdx();

        uint32_t blockNum =
            AscendC::GetBlockNum();

        if (blockNum == 0) {
            return;
        }

        // =====================================================
        // 32 Byte 对齐
        // =====================================================
        constexpr uint32_t ALIGN_ELEMS =
            32 / sizeof(DT_INPUT_X);

        // =====================================================
        // 完整对齐数据
        // =====================================================
        uint32_t alignedLength =
            (totalLength /
             ALIGN_ELEMS)
            * ALIGN_ELEMS;

        uint32_t alignedBlocks =
            alignedLength /
            ALIGN_ELEMS;

        // =====================================================
        // 按对齐 Block 分配到 Core
        // =====================================================
        uint32_t blockStart =
            (alignedBlocks *
             blockIdx)
            / blockNum;

        uint32_t blockEnd =
            (alignedBlocks *
             (blockIdx + 1))
            / blockNum;

        uint32_t start =
            blockStart *
            ALIGN_ELEMS;

        uint32_t end =
            blockEnd *
            ALIGN_ELEMS;

        endLength =
            end - start;

        // =====================================================
        // 对齐数据
        // =====================================================
        if (start < end) {

            uint32_t dataLength =
                end - start;

            uint32_t tileCount =
                (dataLength +
                 tileLength -
                 1)
                / tileLength;

            // =================================================
            // 先预取两个 Tile
            // =================================================
            uint32_t preloadCount =
                tileCount < 2
                    ? tileCount
                    : 2;

            for (uint32_t i = 0;
                 i < preloadCount;
                 ++i) {

                CopyIn(
                    i,
                    start);
            }

            // =================================================
            // 主流水
            // =================================================
            for (uint32_t i = 0;
                 i < tileCount;
                 ++i) {

                // ---------------------------------------------
                // Compute
                // ---------------------------------------------
                Compute(i);

                // ---------------------------------------------
                // 预取后续 Tile
                // ---------------------------------------------
                uint32_t nextTile =
                    i + preloadCount;

                if (nextTile < tileCount) {

                    CopyIn(
                        nextTile,
                        start);
                }

                // ---------------------------------------------
                // CopyOut
                // ---------------------------------------------
                CopyOut(
                    i,
                    start);
            }
        }

        // =====================================================
        // Tail
        // =====================================================
        if (blockIdx == blockNum - 1 &&
            alignedLength < totalLength) {

            uint32_t tailStart =
                alignedLength;

            uint32_t tailLength =
                totalLength -
                alignedLength;

            // =================================================
            // Local Tensor
            // =================================================
            AscendC::LocalTensor<DT_INPUT_X>
                srcLocal =
                    inputQueue
                    .AllocTensor<DT_INPUT_X>();

            AscendC::LocalTensor<DT_INPUT_X>
                dstLocal =
                    outputQueue
                    .AllocTensor<DT_INPUT_X>();

            // =================================================
            // 清零一个完整 32 Byte Block
            // =================================================
            DT_INPUT_X zero =
                static_cast<DT_INPUT_X>(
                    0);

            AscendC::Duplicate(
                srcLocal,
                zero,
                ALIGN_ELEMS);

            // =================================================
            // 读取真正尾部
            // =================================================
            for (uint32_t i = 0;
                 i < tailLength;
                 ++i) {

                DT_INPUT_X value =
                    inputGlobal.GetValue(
                        tailStart + i);

                srcLocal.SetValue(
                    i,
                    value);
            }

            // =================================================
            // GELU
            // =================================================
            ComputeGelu(
                srcLocal,
                dstLocal,
                ALIGN_ELEMS);

            // =================================================
            // 写回真正数据
            // =================================================
            for (uint32_t i = 0;
                 i < tailLength;
                 ++i) {

                DT_INPUT_X result =
                    dstLocal.GetValue(i);

                outputGlobal.SetValue(
                    tailStart + i,
                    result);
            }

            // =================================================
            // Release
            // =================================================
            inputQueue.FreeTensor(
                srcLocal);

            outputQueue.FreeTensor(
                dstLocal);

            // =================================================
            // Flush DCache
            // =================================================
            AscendC::DataCacheCleanAndInvalid<
                DT_INPUT_X,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(
                    outputGlobal[tailStart]);
        }
    }


private:

    // =========================================================
    // Pipe
    // =========================================================
    AscendC::TPipe pipe;

    // =========================================================
    // Input Queue
    // =========================================================
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        2>
        inputQueue;

    // =========================================================
    // Output Queue
    // =========================================================
    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        2>
        outputQueue;

    // =========================================================
    // Erf factor
    // =========================================================
    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        factorBuffer;

    // =========================================================
    // Global Tensor
    // =========================================================
    AscendC::GlobalTensor<DT_INPUT_X>
        inputGlobal;

    AscendC::GlobalTensor<DT_INPUT_X>
        outputGlobal;

    // =========================================================
    // Tiling Parameters
    // =========================================================
    uint32_t totalLength = 0;

    uint32_t tileLength = 0;

    uint32_t endLength = 0;
};


// =================================================================
// Kernel Entry
// =================================================================
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tiling_data,
        tiling);

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tiling_data.length,
        tiling_data.tileLength);

    op.Process();
}