// Kernel侧 GELU 实现

#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"


template <class DT_INPUT_X>
class KernelGelu {

public:

    __aicore__ inline KernelGelu()
    {
    }


    // ============================================================
    // Init
    // ============================================================
    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length,
        uint32_t tileLength)
    {
        length_ = length;
        tileLength_ = tileLength;

        // --------------------------------------------------------
        // GM Tensor
        // --------------------------------------------------------
        inputGm_.SetGlobalBuffer(
            reinterpret_cast<
                __gm__ DT_INPUT_X *
            >(input_x),
            length_);

        outputGm_.SetGlobalBuffer(
            reinterpret_cast<
                __gm__ DT_INPUT_X *
            >(output),
            length_);

        // --------------------------------------------------------
        // Input UB
        // --------------------------------------------------------
        pipe_.InitBuffer(
            inQueue_,
            1,
            tileLength_ *
            sizeof(DT_INPUT_X));

        // --------------------------------------------------------
        // Output UB
        // --------------------------------------------------------
        pipe_.InitBuffer(
            outQueue_,
            1,
            tileLength_ *
            sizeof(DT_INPUT_X));
    }


    // ============================================================
    // Process
    // ============================================================
    __aicore__ inline void Process()
    {
        const uint32_t blockIdx =
            static_cast<uint32_t>(
                AscendC::GetBlockIdx());

        const uint32_t blockNum =
            static_cast<uint32_t>(
                AscendC::GetBlockNum());

        // --------------------------------------------------------
        // 32B 对齐元素数量
        //
        // float16 -> 16
        // float32 -> 8
        // --------------------------------------------------------
        const uint32_t alignElements =
            32 / sizeof(DT_INPUT_X);

        // --------------------------------------------------------
        // 完整的 32B 数据部分
        // --------------------------------------------------------
        const uint32_t alignedLength =
            (length_ / alignElements) *
            alignElements;

        uint32_t start = 0;
        uint32_t end = 0;

        // ========================================================
        // Case 1:
        // 整个 Tensor 小于一个 32B
        // ========================================================
        if (alignedLength == 0) {

            if (blockIdx != 0) {
                return;
            }

            start = 0;
            end = length_;
        }

        // ========================================================
        // Case 2:
        // 正常情况
        // ========================================================
        else {

            // ----------------------------------------------------
            // 只按照完整 32B block 分配 Core
            // ----------------------------------------------------
            const uint32_t totalBlocks =
                alignedLength /
                alignElements;

            const uint32_t blockStart =
                static_cast<uint32_t>(
                    (static_cast<uint64_t>(
                        totalBlocks) *
                     blockIdx) /
                    blockNum);

            const uint32_t blockEnd =
                static_cast<uint32_t>(
                    (static_cast<uint64_t>(
                        totalBlocks) *
                     (blockIdx + 1)) /
                    blockNum);

            start =
                blockStart *
                alignElements;

            end =
                blockEnd *
                alignElements;

            // ----------------------------------------------------
            // 最后一个 Core 负责尾部非对齐数据
            // ----------------------------------------------------
            if (blockIdx + 1 == blockNum) {
                end = length_;
            }
        }

        // ========================================================
        // Tile Loop
        // ========================================================
        uint32_t offset = start;

        while (offset < end) {

            uint32_t currentLength =
                end - offset;

            if (currentLength > tileLength_) {
                currentLength = tileLength_;
            }

            CopyIn(
                offset,
                currentLength);

            Compute(
                currentLength);

            CopyOut(
                offset,
                currentLength);

            offset += currentLength;
        }
    }


private:

    // ============================================================
    // CopyIn
    //
    // 非 32B 对齐尾块：
    //
    // GM
    // 100 elements
    //      ↓
    // DataCopyPad
    //      ↓
    // UB
    // 112 elements
    // ============================================================
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_INPUT_X>
            inputLocal =
                inQueue_.AllocTensor<
                    DT_INPUT_X>();

        const uint32_t alignElements =
            32 / sizeof(DT_INPUT_X);

        const bool aligned =
            ((offset %
              alignElements) == 0) &&
            ((currentLength %
              alignElements) == 0);

        // ========================================================
        // 对齐情况
        // ========================================================
        if (aligned) {

            AscendC::DataCopy(
                inputLocal,
                inputGm_[offset],
                currentLength);
        }

        // ========================================================
        // 非对齐情况
        // ========================================================
        else {

            const uint32_t paddedLength =
                ((currentLength +
                  alignElements - 1) /
                 alignElements) *
                alignElements;

            const uint8_t rightPadding =
                static_cast<uint8_t>(
                    paddedLength -
                    currentLength);

            AscendC::DataCopyExtParams
                copyParams = {
                    1,
                    static_cast<uint32_t>(
                        currentLength *
                        sizeof(DT_INPUT_X)),
                    0,
                    0,
                    0
                };

            AscendC::DataCopyPadExtParams<
                DT_INPUT_X
            > padParams = {
                true,
                0,
                rightPadding,
                static_cast<DT_INPUT_X>(0)
            };

            AscendC::DataCopyPad<DT_INPUT_X>(
                inputLocal,
                inputGm_[offset],
                copyParams,
                padParams);
        }

        inQueue_.EnQue(
            inputLocal);
    }


    // ============================================================
    // Compute
    //
    // GELU:
    //
    //     GELU(x)
    //       = x * Phi(x)
    //
    //       = 0.5 * x *
    //         (1 + erf(x / sqrt(2)))
    //
    // 这里不再调用 AscendC::Gelu，
    // 而是严格按照题目公式展开。
    // ============================================================
    __aicore__ inline void Compute(
        uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_INPUT_X>
            inputLocal =
                inQueue_.DeQue<
                    DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X>
            outputLocal =
                outQueue_.AllocTensor<
                    DT_INPUT_X>();

        const uint32_t alignElements =
            32 / sizeof(DT_INPUT_X);

        // --------------------------------------------------------
        // 非对齐尾块需要按照 32B 对齐长度计算
        // --------------------------------------------------------
        const uint32_t paddedLength =
            ((currentLength +
              alignElements - 1) /
             alignElements) *
            alignElements;

        // --------------------------------------------------------
        // 常数
        //
        // 1 / sqrt(2)
        // --------------------------------------------------------
        const DT_INPUT_X invSqrt2 =
            static_cast<DT_INPUT_X>(
                0.70710678118654752440);

        const DT_INPUT_X one =
            static_cast<DT_INPUT_X>(1.0);

        const DT_INPUT_X half =
            static_cast<DT_INPUT_X>(0.5);

        // ========================================================
        // 1.
        //
        // tmp = x / sqrt(2)
        //
        // 直接修改 inputLocal。
        //
        // 注意：
        // 后面最终还需要原始 x，
        // 因此这里不能修改 inputLocal。
        //
        // 所以采用 outputLocal 暂存。
        // ========================================================

        AscendC::Muls(
            outputLocal,
            inputLocal,
            invSqrt2,
            paddedLength);

        // ========================================================
        // 2.
        //
        // output = erf(x / sqrt(2))
        // ========================================================
        AscendC::Erf(
            outputLocal,
            outputLocal,
            paddedLength);

        // ========================================================
        // 3.
        //
        // output =
        //     1 + erf(x / sqrt(2))
        // ========================================================
        AscendC::Adds(
            outputLocal,
            outputLocal,
            one,
            paddedLength);

        // ========================================================
        // 4.
        //
        // output =
        //     0.5 * (1 + erf(...))
        // ========================================================
        AscendC::Muls(
            outputLocal,
            outputLocal,
            half,
            paddedLength);

        // ========================================================
        // 5.
        //
        // output =
        //     x * 0.5 * (1 + erf(...))
        // ========================================================
        //
        // 注意：
        // inputLocal 仍然保存原始 x。
        //
        AscendC::Mul(
            outputLocal,
            outputLocal,
            inputLocal,
            paddedLength);

        // ========================================================
        // Enqueue output
        // ========================================================
        outQueue_.EnQue<
            DT_INPUT_X>(
                outputLocal);

        inQueue_.FreeTensor(
            inputLocal);
    }


    // ============================================================
    // CopyOut
    //
    // 对于尾块：
    //
    // UB:
    //     paddedLength
    //
    // GM:
    //     currentLength
    //
    // 只写真实数据。
    // ============================================================
    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<DT_INPUT_X>
            outputLocal =
                outQueue_.DeQue<
                    DT_INPUT_X>();

        const uint32_t alignElements =
            32 / sizeof(DT_INPUT_X);

        const bool aligned =
            ((offset %
              alignElements) == 0) &&
            ((currentLength %
              alignElements) == 0);

        // ========================================================
        // 对齐
        // ========================================================
        if (aligned) {

            AscendC::DataCopy(
                outputGm_[offset],
                outputLocal,
                currentLength);
        }

        // ========================================================
        // 非对齐尾块
        // ========================================================
        else {

            AscendC::DataCopyExtParams
                copyParams = {
                    1,
                    static_cast<uint32_t>(
                        currentLength *
                        sizeof(DT_INPUT_X)),
                    0,
                    0,
                    0
                };

            AscendC::DataCopyPad<DT_INPUT_X>(
                outputGm_[offset],
                outputLocal,
                copyParams);
        }

        outQueue_.FreeTensor(
            outputLocal);
    }


private:

    // ============================================================
    // GM
    // ============================================================
    AscendC::GlobalTensor<DT_INPUT_X>
        inputGm_;

    AscendC::GlobalTensor<DT_INPUT_X>
        outputGm_;

    // ============================================================
    // Pipe
    // ============================================================
    AscendC::TPipe pipe_;

    // ============================================================
    // Input Queue
    // ============================================================
    AscendC::TQue<
        AscendC::TPosition::VECIN,
        1
    > inQueue_;

    // ============================================================
    // Output Queue
    // ============================================================
    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        1
    > outQueue_;

    // ============================================================
    // Tiling
    // ============================================================
    uint32_t length_ = 0;

    uint32_t tileLength_ = 0;
};


// ============================================================================
// Kernel Entry
// ============================================================================

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