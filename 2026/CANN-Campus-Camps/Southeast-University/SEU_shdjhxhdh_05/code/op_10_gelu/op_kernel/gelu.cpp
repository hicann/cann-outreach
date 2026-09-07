/*!
 * \file gelu.cpp
 * \brief GELU 算子 Kernel
 */

#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"


template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu()
    {
    }

    // =========================================================
    // Init
    // =========================================================
    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t blockLength,
        uint32_t lastBlockLength,
        uint32_t tileLength)
    {
        const uint32_t blockIdx =
            static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t blockNum =
            static_cast<uint32_t>(AscendC::GetBlockNum());

        // -----------------------------------------------------
        // 当前 Core 实际处理长度
        // -----------------------------------------------------
        if (blockIdx == blockNum - 1U) {
            currentLength = lastBlockLength;
        } else {
            currentLength = blockLength;
        }

        // -----------------------------------------------------
        // 当前 Core 在 GM 中的起始位置
        // -----------------------------------------------------
        const uint32_t offset =
            blockIdx * blockLength;

        // -----------------------------------------------------
        // 设置 GM Buffer
        // -----------------------------------------------------
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(input_x) + offset,
            currentLength);

        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(output) + offset,
            currentLength);

        this->tileLength = tileLength;

        // -----------------------------------------------------
        // Input / Output Queue
        // -----------------------------------------------------
        pipe.InitBuffer(
            inQueueX,
            2,
            this->tileLength * sizeof(T));

        pipe.InitBuffer(
            outQueueY,
            2,
            this->tileLength * sizeof(T));

        // -----------------------------------------------------
        // GELU 中间结果：
        //
        // tmp = x / sqrt(2)
        //
        // 使用 VECCALC 保存中间结果
        // -----------------------------------------------------
        pipe.InitBuffer(
            tmpQueue,
            2,
            this->tileLength * sizeof(T));
    }


    // =========================================================
    // Process
    // =========================================================
__aicore__ inline void Process()
{
    uint32_t offset = 0U;
    while (offset < currentLength) {
        uint32_t currentTileLength = currentLength - offset;
        if (currentTileLength > tileLength) {
            currentTileLength = tileLength;
        }
        CopyIn(offset, currentTileLength);
        Compute(currentTileLength);
        CopyOut(offset, currentTileLength);
        offset += currentTileLength;
    }
}


private:

    // =========================================================
    // CopyIn
    // =========================================================
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t dataLength)
    {
        AscendC::LocalTensor<T> xLocal =
            inQueueX.AllocTensor<T>();

        const uint32_t copyBytes =
            dataLength * sizeof(T);

        // -----------------------------------------------------
        // 32 Byte 对齐时直接 DataCopy
        // -----------------------------------------------------
        if ((copyBytes % 32U) == 0U) {

            AscendC::DataCopy(
                xLocal,
                inputGm[offset],
                dataLength);

        }
        // -----------------------------------------------------
        // 非 32 Byte 对齐时使用 DataCopyPad
        // -----------------------------------------------------
        else {

            AscendC::DataCopyExtParams copyParams = {
                1,
                copyBytes,
                0,
                0,
                0
            };

            AscendC::DataCopyPadExtParams<T> padParams = {
                true,
                0,
                0,
                static_cast<T>(0)
            };

            AscendC::DataCopyPad(
                xLocal,
                inputGm[offset],
                copyParams,
                padParams);
        }

        inQueueX.EnQue<T>(xLocal);
    }


    // =========================================================
    // Compute
    //
    // GELU(x)
    //
    // = x * 0.5 * (1 + erf(x / sqrt(2)))
    // =========================================================
    __aicore__ inline void Compute(
        uint32_t dataLength)
    {
        // -----------------------------------------------------
        // 取输入
        // -----------------------------------------------------
        AscendC::LocalTensor<T> xLocal =
            inQueueX.DeQue<T>();

        // -----------------------------------------------------
        // 输出 Tensor
        // -----------------------------------------------------
        AscendC::LocalTensor<T> yLocal =
            outQueueY.AllocTensor<T>();

        // -----------------------------------------------------
        // 临时 Tensor
        //
        // tmpLocal:
        //     x / sqrt(2)
        // -----------------------------------------------------
        AscendC::LocalTensor<T> tmpLocal =
            tmpQueue.AllocTensor<T>();


        // =====================================================
        // 常量
        // =====================================================

        // 1 / sqrt(2)
        constexpr float INV_SQRT2 =
            0.70710678118654752440f;

        // 0.5
        constexpr float HALF =
            0.5f;




        AscendC::Muls(
            tmpLocal,
            xLocal,
            static_cast<T>(INV_SQRT2),
            static_cast<int32_t>(dataLength));




// y = erf(x / sqrt(2))
AscendC::Erf<T, false>(
    yLocal,
    tmpLocal,
    dataLength);

// y = x * erf(x / sqrt(2)) + x
AscendC::FusedMulAdd(
    yLocal,
    xLocal,
    xLocal,
    static_cast<int32_t>(dataLength));

// y = 0.5 * (x * erf(x / sqrt(2)) + x)
AscendC::Muls(
    yLocal,
    yLocal,
    static_cast<T>(HALF),
    static_cast<int32_t>(dataLength));


        // -----------------------------------------------------
        // 释放输入
        // -----------------------------------------------------
        inQueueX.FreeTensor(xLocal);

        // -----------------------------------------------------
        // 释放临时 Tensor
        // -----------------------------------------------------
        tmpQueue.FreeTensor(tmpLocal);

        // -----------------------------------------------------
        // 输出进入 Queue
        // -----------------------------------------------------
        outQueueY.EnQue<T>(yLocal);
    }


    // =========================================================
    // CopyOut
    // =========================================================
    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t dataLength)
    {
        AscendC::LocalTensor<T> yLocal =
            outQueueY.DeQue<T>();

        const uint32_t copyBytes =
            dataLength * sizeof(T);

        // -----------------------------------------------------
        // 32 Byte 对齐
        // -----------------------------------------------------
        if ((copyBytes % 32U) == 0U) {

            AscendC::DataCopy(
                outputGm[offset],
                yLocal,
                dataLength);
        }

        // -----------------------------------------------------
        // 非 32 Byte 对齐
        // -----------------------------------------------------
        else {

            AscendC::DataCopyExtParams copyParams = {
                1,
                copyBytes,
                0,
                0,
                0
            };

            AscendC::DataCopyPad(
                outputGm[offset],
                yLocal,
                copyParams);
        }

        outQueueY.FreeTensor(yLocal);
    }


private:

    // =========================================================
    // Pipe
    // =========================================================

    AscendC::TPipe pipe;

    // GM Input
    AscendC::GlobalTensor<T> inputGm;

    // GM Output
    AscendC::GlobalTensor<T> outputGm;

    // ---------------------------------------------------------
    // Input Queue
    // ---------------------------------------------------------
    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> inQueueX;

    // ---------------------------------------------------------
    // Output Queue
    // ---------------------------------------------------------
    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> outQueueY;

    // ---------------------------------------------------------
    // Temporary Queue
    // ---------------------------------------------------------
    AscendC::TQue<
        AscendC::QuePosition::VECCALC,
        1> tmpQueue;

    // 当前 Core 处理的数据长度
    uint32_t currentLength = 0U;

    // 每次 Tile 大小
    uint32_t tileLength = 0U;
};


// =============================================================
// Kernel Entry
// =============================================================

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tiling_data,
        tiling);

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tiling_data.blockLength,
        tiling_data.lastBlockLength,
        tiling_data.tileLength);

    op.Process();
}