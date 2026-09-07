// Kernel侧核函数实现
#include "kernel_operator.h"
#include "lib/math/erf.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                               const GeluTilingData &tiling) {
        tileLength_ = tiling.tileLength;
        const uint64_t totalTiles = tiling.length / tileLength_ +
            (tiling.length % tileLength_ != 0);
        const uint64_t tilesPerCore = totalTiles / tiling.coreCount;
        const uint32_t extraTiles = totalTiles % tiling.coreCount;
        const uint32_t core = GetBlockIdx();
        const uint64_t firstTile = core * tilesPerCore +
            (core < extraTiles ? core : extraTiles);
        const uint64_t coreTiles = tilesPerCore + (core < extraTiles);
        const uint64_t offset = firstTile * tileLength_;
        if (offset >= tiling.length || coreTiles == 0) {
            blockLength_ = 0;
            return;
        }
        const uint64_t remaining = tiling.length - offset;
        blockLength_ = coreTiles * tileLength_ < remaining
            ? coreTiles * tileLength_ : remaining;
        inputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + offset, blockLength_);
        outputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + offset, blockLength_);
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(scaledBuffer_, tileLength_ * sizeof(float));
        if constexpr (sizeof(DT_INPUT_X) != sizeof(float)) {
            pipe_.InitBuffer(xBuffer_, tileLength_ * sizeof(float));
            pipe_.InitBuffer(erfBuffer_, tileLength_ * sizeof(float));
        }
        // Erf<float> uses three disjoint float32 temporary tensors.
        pipe_.InitBuffer(erfTmpBuffer_, 3 * tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0) {
            return;
        }
        if (blockLength_ <= tileLength_) {
            const uint32_t count = static_cast<uint32_t>(blockLength_);
            CopyIn(0, count);
            Compute(count);
            CopyOut(0, count);
            return;
        }

        CopyIn(0, tileLength_);
        uint32_t count = tileLength_;
        for (uint64_t offset = 0; offset < blockLength_; offset += tileLength_) {
            const uint64_t nextOffset = offset + tileLength_;
            uint32_t nextCount = 0;
            if (nextOffset < blockLength_) {
                const uint64_t remaining = blockLength_ - nextOffset;
                nextCount = remaining < tileLength_
                    ? static_cast<uint32_t>(remaining) : tileLength_;
                // Queue at most two inputs; MTE2 can load the next tile while V computes this one.
                CopyIn(nextOffset, nextCount);
            }
            Compute(count);
            CopyOut(offset, count);
            count = nextCount;
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count) {
        LocalTensor<DT_INPUT_X> input = inputQueue_.AllocTensor<DT_INPUT_X>();
        if (count % (32 / sizeof(DT_INPUT_X)) == 0) {
            DataCopy(input, inputGm_[offset], count);
        } else {
            const DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
            const uint32_t alignElements = 32 / sizeof(DT_INPUT_X);
            const uint8_t rightPadding = static_cast<uint8_t>(
                alignElements - count % alignElements);
            const DataCopyPadExtParams<DT_INPUT_X> padding{
                true, 0, rightPadding, static_cast<DT_INPUT_X>(0)};
            DataCopyPad(input, inputGm_[offset], params, padding);
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_INPUT_X> input = inputQueue_.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> output = outputQueue_.AllocTensor<DT_INPUT_X>();
        LocalTensor<float> x;
        LocalTensor<float> result;
        if constexpr (sizeof(DT_INPUT_X) == sizeof(float)) {
            x = input.template ReinterpretCast<float>();
            result = output.template ReinterpretCast<float>();
        } else {
            x = xBuffer_.Get<float>();
            result = erfBuffer_.Get<float>();
            Cast(x, input, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> scaled = scaledBuffer_.Get<float>();
        LocalTensor<uint8_t> erfTmp = erfTmpBuffer_.Get<uint8_t>();

        // Match torch.nn.functional.gelu(..., approximate="none").
        // Erf requires non-overlapping source, destination and temporary buffers.
        Muls(scaled, x, 0.7071067811865475f, count);
        PipeBarrier<PIPE_V>();
        Erf(result, scaled, erfTmp, count);
        PipeBarrier<PIPE_V>();
        Maxs(result, result, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Mins(result, result, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(result, result, 1.0f, count);
        PipeBarrier<PIPE_V>();
        // Scale the CDF first so positive finite inputs cannot overflow at 2*x.
        Muls(result, result, 0.5f, count);
        PipeBarrier<PIPE_V>();
        Mul(result, x, result, count);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(DT_INPUT_X) != sizeof(float)) {
            // Use the float32-to-half mode used by the CANN Erf implementation on 910B.
            Cast(output, result, RoundMode::CAST_ROUND, count);
            PipeBarrier<PIPE_V>();
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count) {
        LocalTensor<DT_INPUT_X> output = outputQueue_.DeQue<DT_INPUT_X>();
        if (count % (32 / sizeof(DT_INPUT_X)) == 0) {
            DataCopy(outputGm_[offset], output, count);
        } else {
            const DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], output, params);
        }
        outputQueue_.FreeTensor(output);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<TPosition::VECCALC> xBuffer_;
    TBuf<TPosition::VECCALC> scaledBuffer_;
    TBuf<TPosition::VECCALC> erfBuffer_;
    TBuf<TPosition::VECCALC> erfTmpBuffer_;
    GlobalTensor<DT_INPUT_X> inputGm_;
    GlobalTensor<DT_INPUT_X> outputGm_;
    uint64_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
