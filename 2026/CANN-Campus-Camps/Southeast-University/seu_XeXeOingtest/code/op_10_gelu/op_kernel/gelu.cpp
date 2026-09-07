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
        // Input Queue
        // =====================================================
        pipe.InitBuffer(
            inputQueue,
            1,
            tileLength *
            sizeof(DT_INPUT_X));

        // =====================================================
        // Output Queue
        // =====================================================
        pipe.InitBuffer(
            outputQueue,
            1,
            tileLength *
            sizeof(DT_INPUT_X));

        // =====================================================
        // Temporary Buffer
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
    //
    // = 0.5 * x *
    //   (1 + erf(x / sqrt(2)))
    //
    // 对应 PyTorch:
    //
    // torch.nn.functional.gelu(x)
    //
    // 默认 approximate='none'
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
        // 1. dst = x / sqrt(2)
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
        // 2. factor = erf(x / sqrt(2))
        // =====================================================
        AscendC::Erf(
            factorLocal,
            dstLocal,
            count);

        // =====================================================
        // 3. dst = 1 + erf(...)
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
        // 4. factor = 0.5 * (1 + erf(...))
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
        // 5. dst = x * factor
        // =====================================================
        AscendC::Mul(
            dstLocal,
            srcLocal,
            factorLocal,
            count);
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
        //
        // FP16 -> 16 elements
        // FP32 -> 8 elements
        // =====================================================
        constexpr uint32_t ALIGN_ELEMS =
            32 / sizeof(DT_INPUT_X);

        // =====================================================
        // 当前只有一个 Core
        // =====================================================
        uint32_t start = 0;
        uint32_t end = totalLength;

        // =====================================================
        // 计算完整对齐部分
        // =====================================================
        uint32_t dataLength =
            end - start;

        uint32_t alignedLength =
            (dataLength / ALIGN_ELEMS)
            * ALIGN_ELEMS;

        uint32_t pos = start;

        // =====================================================
        // 处理完整的 32 Byte 对齐部分
        // =====================================================
        while (pos < alignedLength) {

            uint32_t currentLength =
                alignedLength - pos;

            if (currentLength > tileLength) {
                currentLength = tileLength;
            }

            // -------------------------------------------------
            // 由于 tileLength 已经经过 32 Byte 对齐，
            // currentLength 在对齐区域中也保持对齐。
            // -------------------------------------------------

            // =================================================
            // GM -> UB
            // =================================================
            AscendC::LocalTensor<DT_INPUT_X>
                srcLocal =
                    inputQueue.AllocTensor<DT_INPUT_X>();

            AscendC::DataCopy(
                srcLocal,
                inputGlobal[pos],
                currentLength);

            inputQueue.EnQue(srcLocal);

            // =================================================
            // UB -> Compute
            // =================================================
            srcLocal =
                inputQueue.DeQue<DT_INPUT_X>();

            AscendC::LocalTensor<DT_INPUT_X>
                dstLocal =
                    outputQueue.AllocTensor<DT_INPUT_X>();

            ComputeGelu(
                srcLocal,
                dstLocal,
                currentLength);

            outputQueue.EnQue(dstLocal);

            inputQueue.FreeTensor(srcLocal);

            // =================================================
            // UB -> GM
            // =================================================
            dstLocal =
                outputQueue.DeQue<DT_INPUT_X>();

            AscendC::DataCopy(
                outputGlobal[pos],
                dstLocal,
                currentLength);

            outputQueue.FreeTensor(dstLocal);

            pos += currentLength;
        }

        // =====================================================
        // Tail
        //
        // 处理不足一个 32 Byte 的最后几个元素。
        //
        // FP16:
        //     最多 15 个元素
        //
        // FP32:
        //     最多 7 个元素
        // =====================================================
        if (pos < end) {

            uint32_t tailLength =
                end - pos;

            // =================================================
            // 分配 Local Tensor
            // =================================================
            AscendC::LocalTensor<DT_INPUT_X>
                srcLocal =
                    inputQueue.AllocTensor<DT_INPUT_X>();

            AscendC::LocalTensor<DT_INPUT_X>
                dstLocal =
                    outputQueue.AllocTensor<DT_INPUT_X>();

            // =================================================
            // 先将一个完整 32 Byte Block 清零
            // =================================================
            DT_INPUT_X zero =
                static_cast<DT_INPUT_X>(
                    0);

            AscendC::Duplicate(
                srcLocal,
                zero,
                ALIGN_ELEMS);

            // =================================================
            // GM -> Local
            //
            // 由于尾部不满足 DataCopy 对齐要求，
            // 使用逐元素读取。
            // =================================================
            for (uint32_t i = 0;
                 i < tailLength;
                 ++i) {

                DT_INPUT_X value =
                    inputGlobal.GetValue(
                        pos + i);

                srcLocal.SetValue(
                    i,
                    value);
            }

            // =================================================
            // 对整个对齐 Block 做 GELU
            //
            // 补零部分:
            //
            // GELU(0) = 0
            //
            // 不影响真正有效的数据。
            // =================================================
            ComputeGelu(
                srcLocal,
                dstLocal,
                ALIGN_ELEMS);

            // =================================================
            // Local -> GM
            //
            // 只写回真实存在的尾部元素。
            // =================================================
            for (uint32_t i = 0;
                 i < tailLength;
                 ++i) {

                DT_INPUT_X result =
                    dstLocal.GetValue(i);

                outputGlobal.SetValue(
                    pos + i,
                    result);
            }

            // =================================================
            // 释放 Local Tensor
            // =================================================
            inputQueue.FreeTensor(
                srcLocal);

            outputQueue.FreeTensor(
                dstLocal);

            // =================================================
            // 刷新尾部 DCache
            // =================================================
            AscendC::DataCacheCleanAndInvalid<
                DT_INPUT_X,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(
                    outputGlobal[pos]);
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
        1> inputQueue;

    // =========================================================
    // Output Queue
    // =========================================================
    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        1> outputQueue;

    // =========================================================
    // Temporary Buffer
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