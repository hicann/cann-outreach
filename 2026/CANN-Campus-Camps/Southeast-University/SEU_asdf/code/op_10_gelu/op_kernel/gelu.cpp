#include "kernel_operator.h"
#include <type_traits>

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

template <class DT_INPUT_X>
class KernelGeluV3 {
public:
    __aicore__ inline KernelGeluV3() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                uint32_t length, uint32_t blockLength,
                                AscendC::TPipe *pipe) {
        totalLength_ = length;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t start = blockIdx * blockLength;

        if (start >= length) {
            blockLength_ = 0U;
            return;
        }

        blockLength_ = (start + blockLength <= length) ? blockLength : (length - start);
        blockStartAligned_ = ((start * sizeof(DT_INPUT_X)) & 31U) == 0U;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + start,
            blockLength_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(output) + start,
            blockLength_);

        const uint8_t bufferNum =
            (totalLength_ >= LARGE_LENGTH_THRESHOLD && blockLength_ > TILE_ELEMS) ? 2U : 1U;

        pipe->InitBuffer(inputQueue_, bufferNum, TILE_BYTES);
        pipe->InitBuffer(outputQueue_, bufferNum, TILE_BYTES);

        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            pipe->InitBuffer(tmpFloat0_, TILE_ELEMS * sizeof(float));
            pipe->InitBuffer(tmpFloat1_, TILE_ELEMS * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0U) {
            return;
        }

        if (blockLength_ <= TILE_ELEMS) {
            CopyIn(0U, blockLength_);
            Compute(blockLength_);
            CopyOut(0U, blockLength_);
            return;
        }

        uint32_t offset = 0U;
        while (offset < blockLength_) {
            const uint32_t remain = blockLength_ - offset;
            const uint32_t count = (remain > TILE_ELEMS) ? TILE_ELEMS : remain;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline bool CanUseAlignedCopy(uint32_t count) const {
        return blockStartAligned_ &&
               (((count * sizeof(DT_INPUT_X)) & 31U) == 0U);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal =
            inputQueue_.AllocTensor<DT_INPUT_X>();

        if (CanUseAlignedCopy(count)) {
            AscendC::DataCopy(inputLocal, inputGm_[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(count * sizeof(DT_INPUT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{
                false,
                0,
                0,
                static_cast<DT_INPUT_X>(0)
            };
            AscendC::DataCopyPad(
                inputLocal,
                inputGm_[offset],
                copyParams,
                padParams);
        }

        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal =
            inputQueue_.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> outputLocal =
            outputQueue_.AllocTensor<DT_INPUT_X>();

        constexpr float INV_SQRT2 = 0.70710678118654752440f;

        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            AscendC::LocalTensor<float> tmp0 = tmpFloat0_.Get<float>();
            AscendC::LocalTensor<float> tmp1 = tmpFloat1_.Get<float>();

            AscendC::Cast(
                tmp0,
                inputLocal,
                AscendC::RoundMode::CAST_NONE,
                count);

            AscendC::Muls(
                tmp1,
                tmp0,
                INV_SQRT2,
                count);

            AscendC::Erf<float, false>(
                tmp0,
                tmp1,
                count);

            AscendC::Adds(
                tmp0,
                tmp0,
                1.0f,
                count);

            AscendC::Mul(
                tmp1,
                tmp1,
                tmp0,
                count);

            AscendC::Muls(
                tmp1,
                tmp1,
                INV_SQRT2,
                count);

            AscendC::Cast(
                outputLocal,
                tmp1,
                AscendC::RoundMode::CAST_RINT,
                count);
        } else {
            AscendC::Muls(
                outputLocal,
                inputLocal,
                INV_SQRT2,
                count);

            AscendC::Erf<float, false>(
                inputLocal,
                outputLocal,
                count);

            AscendC::Adds(
                inputLocal,
                inputLocal,
                1.0f,
                count);

            AscendC::Mul(
                outputLocal,
                outputLocal,
                inputLocal,
                count);

            AscendC::Muls(
                outputLocal,
                outputLocal,
                INV_SQRT2,
                count);
        }

        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_INPUT_X> outputLocal =
            outputQueue_.DeQue<DT_INPUT_X>();

        if (CanUseAlignedCopy(count)) {
            AscendC::DataCopy(outputGm_[offset], outputLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(count * sizeof(DT_INPUT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(
                outputGm_[offset],
                outputLocal,
                copyParams);
        }

        outputQueue_.FreeTensor(outputLocal);
    }

private:
    static constexpr uint32_t TILE_BYTES =
        std::is_same_v<DT_INPUT_X, half>
            ? (16U * 1024U)
            : (32U * 1024U);

    static constexpr uint32_t TILE_ELEMS =
        TILE_BYTES / sizeof(DT_INPUT_X);

    static constexpr uint32_t LARGE_LENGTH_THRESHOLD =
        64U * 1024U;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        2> inputQueue_;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        2> outputQueue_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC> tmpFloat0_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC> tmpFloat1_;

    AscendC::GlobalTensor<DT_INPUT_X> inputGm_;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm_;

    uint32_t totalLength_ = 0U;
    uint32_t blockLength_ = 0U;
    bool blockStartAligned_ = false;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling) {

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tilingData,
        tiling);

    AscendC::TPipe pipe;
    KernelGeluV3<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tilingData.length,
        tilingData.blockLength,
        &pipe);

    op.Process();
}
