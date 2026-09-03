// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"


// ============================================================
// FP32：精确定义 GELU
//
// GELU(x)
// = 0.5 * x *
//   (1 + erf(x / sqrt(2)))
// ============================================================

__aicore__ inline void ComputeExactGelu(
    const AscendC::LocalTensor<float>& outputLocal,
    const AscendC::LocalTensor<float>& inputLocal,
    const AscendC::LocalTensor<float>& tmpX,
    const AscendC::LocalTensor<float>& tmpWork,
    const AscendC::LocalTensor<float>& tmpErf,
    uint32_t length)
{
    constexpr float INV_SQRT_2 =
        0.70710678118654752440f;


    // --------------------------------------------------------
    // tmpWork = x / sqrt(2)
    // --------------------------------------------------------

    AscendC::Muls(
        tmpWork,
        inputLocal,
        INV_SQRT_2,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpErf = erf(x / sqrt(2))
    // --------------------------------------------------------

    AscendC::Erf<float, false>(
        tmpErf,
        tmpWork,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpErf = 1 + erf(...)
    // --------------------------------------------------------

    AscendC::Adds(
        tmpErf,
        tmpErf,
        1.0f,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpWork =
    // x * (1 + erf(...))
    // --------------------------------------------------------

    AscendC::Mul(
        tmpWork,
        inputLocal,
        tmpErf,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // output =
    // 0.5 * x * (1 + erf(...))
    // --------------------------------------------------------

    AscendC::Muls(
        outputLocal,
        tmpWork,
        0.5f,
        length);

    AscendC::PipeBarrier<PIPE_V>();
}


// ============================================================
// FP16：先转FP32，再按erf定义计算，最后转回FP16
// ============================================================

__aicore__ inline void ComputeExactGelu(
    const AscendC::LocalTensor<half>& outputLocal,
    const AscendC::LocalTensor<half>& inputLocal,
    const AscendC::LocalTensor<float>& tmpX,
    const AscendC::LocalTensor<float>& tmpWork,
    const AscendC::LocalTensor<float>& tmpErf,
    uint32_t length)
{
    constexpr float INV_SQRT_2 =
        0.70710678118654752440f;


    // --------------------------------------------------------
    // tmpX = float(x)
    // --------------------------------------------------------

    AscendC::Cast<float, half>(
        tmpX,
        inputLocal,
        AscendC::RoundMode::CAST_NONE,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpWork = x / sqrt(2)
    // --------------------------------------------------------

    AscendC::Muls(
        tmpWork,
        tmpX,
        INV_SQRT_2,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpErf = erf(x / sqrt(2))
    // --------------------------------------------------------

    AscendC::Erf<float, false>(
        tmpErf,
        tmpWork,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpErf = 1 + erf(...)
    // --------------------------------------------------------

    AscendC::Adds(
        tmpErf,
        tmpErf,
        1.0f,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpWork =
    // x * (1 + erf(...))
    // --------------------------------------------------------

    AscendC::Mul(
        tmpWork,
        tmpX,
        tmpErf,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // tmpWork *= 0.5
    // --------------------------------------------------------

    AscendC::Muls(
        tmpWork,
        tmpWork,
        0.5f,
        length);

    AscendC::PipeBarrier<PIPE_V>();


    // --------------------------------------------------------
    // float -> half
    // --------------------------------------------------------

    AscendC::Cast<half, float>(
        outputLocal,
        tmpWork,
        AscendC::RoundMode::CAST_RINT,
        length);

    AscendC::PipeBarrier<PIPE_V>();
}



// ============================================================
// KernelGelu
// ============================================================

template <class DT_INPUT_X>
class KernelGelu {
public:

    __aicore__ inline KernelGelu()
    {
    }


    // ========================================================
    // Init
    // ========================================================

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length)
    {
        totalLength =
            length;


        // ----------------------------------------------------
        // Global Memory
        // ----------------------------------------------------

        inputGm.SetGlobalBuffer(
            (__gm__ DT_INPUT_X*)input_x,
            totalLength);

        outputGm.SetGlobalBuffer(
            (__gm__ DT_INPUT_X*)output,
            totalLength);


        // ----------------------------------------------------
        // 输入Queue
        // ----------------------------------------------------

        pipe.InitBuffer(
            inputQueue,
            1,
            TILE_LENGTH *
                sizeof(DT_INPUT_X));


        // ----------------------------------------------------
        // 输出Queue
        // ----------------------------------------------------

        pipe.InitBuffer(
            outputQueue,
            1,
            TILE_LENGTH *
                sizeof(DT_INPUT_X));


        // ----------------------------------------------------
        // FP32临时空间
        //
        // tmpX:
        //   FP16路径保存转成FP32后的x
        //
        // tmpWork:
        //   x/sqrt(2)
        //   以及后续乘法结果
        //
        // tmpErf:
        //   erf结果
        // ----------------------------------------------------

        pipe.InitBuffer(
            tmpXBuffer,
            TILE_LENGTH *
                sizeof(float));

        pipe.InitBuffer(
            tmpWorkBuffer,
            TILE_LENGTH *
                sizeof(float));

        pipe.InitBuffer(
            tmpErfBuffer,
            TILE_LENGTH *
                sizeof(float));
    }


    // ========================================================
    // Process
    // ========================================================

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;


        while (offset <
               totalLength) {

            uint32_t currentLength =
                totalLength -
                offset;


            if (currentLength >
                TILE_LENGTH) {

                currentLength =
                    TILE_LENGTH;
            }


            CopyIn(
                offset,
                currentLength);


            Compute(
                currentLength);


            CopyOut(
                offset,
                currentLength);


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
        AscendC::LocalTensor<
            DT_INPUT_X>
            inputLocal =
                inputQueue
                    .AllocTensor<
                        DT_INPUT_X>();


        // ----------------------------------------------------
        // blockLen单位为Byte
        // ----------------------------------------------------

        AscendC::DataCopyExtParams
            copyParams{
                1,

                static_cast<uint32_t>(
                    length *
                    sizeof(DT_INPUT_X)),

                0,
                0,
                0
            };


        AscendC::DataCopyPadExtParams<
            DT_INPUT_X>
            padParams{
                false,
                0,
                0,
                0
            };


        AscendC::DataCopyPad(
            inputLocal,
            inputGm[offset],
            copyParams,
            padParams);


        inputQueue.EnQue(
            inputLocal);
    }


    // ========================================================
    // Compute
    // ========================================================

    __aicore__ inline void Compute(
        uint32_t length)
    {
        // ----------------------------------------------------
        // 输入
        // ----------------------------------------------------

        AscendC::LocalTensor<
            DT_INPUT_X>
            inputLocal =
                inputQueue
                    .DeQue<
                        DT_INPUT_X>();


        // ----------------------------------------------------
        // 输出
        // ----------------------------------------------------

        AscendC::LocalTensor<
            DT_INPUT_X>
            outputLocal =
                outputQueue
                    .AllocTensor<
                        DT_INPUT_X>();


        // ----------------------------------------------------
        // 临时FP32 Tensor
        // ----------------------------------------------------

        AscendC::LocalTensor<float>
            tmpX =
                tmpXBuffer
                    .Get<float>();


        AscendC::LocalTensor<float>
            tmpWork =
                tmpWorkBuffer
                    .Get<float>();


        AscendC::LocalTensor<float>
            tmpErf =
                tmpErfBuffer
                    .Get<float>();


        // ----------------------------------------------------
        // 自动匹配float / half重载
        // ----------------------------------------------------

        ComputeExactGelu(
            outputLocal,
            inputLocal,
            tmpX,
            tmpWork,
            tmpErf,
            length);


        // ----------------------------------------------------
        // 输出入队
        // ----------------------------------------------------

        outputQueue.EnQue(
            outputLocal);


        // ----------------------------------------------------
        // 释放输入
        // ----------------------------------------------------

        inputQueue.FreeTensor(
            inputLocal);
    }


    // ========================================================
    // CopyOut
    // ========================================================

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t length)
    {
        AscendC::LocalTensor<
            DT_INPUT_X>
            outputLocal =
                outputQueue
                    .DeQue<
                        DT_INPUT_X>();


        AscendC::DataCopyExtParams
            copyParams{
                1,

                static_cast<uint32_t>(
                    length *
                    sizeof(DT_INPUT_X)),

                0,
                0,
                0
            };


        AscendC::DataCopyPad(
            outputGm[offset],
            outputLocal,
            copyParams);


        outputQueue.FreeTensor(
            outputLocal);
    }


private:

    // ========================================================
    // Erf属于高阶API，会占用额外UB。
    //
    // 先使用512，正确性优先。
    // ========================================================

    static constexpr uint32_t
        TILE_LENGTH = 512;


    // ========================================================
    // Pipe
    // ========================================================

    AscendC::TPipe pipe;


    // ========================================================
    // Queue
    // ========================================================

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        1>
        inputQueue;


    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        1>
        outputQueue;


    // ========================================================
    // FP32 Buffers
    // ========================================================

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        tmpXBuffer;


    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        tmpWorkBuffer;


    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        tmpErfBuffer;


    // ========================================================
    // GM
    // ========================================================

    AscendC::GlobalTensor<
        DT_INPUT_X>
        inputGm;


    AscendC::GlobalTensor<
        DT_INPUT_X>
        outputGm;


    uint32_t totalLength;
};



// ============================================================
// Kernel入口
// ============================================================

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


    KernelGelu<
        DT_INPUT_X>
        op;


    op.Init(
        input_x,
        output,
        tiling_data.length);


    op.Process();


    AscendC::PipeBarrier<
        PIPE_ALL>();
}