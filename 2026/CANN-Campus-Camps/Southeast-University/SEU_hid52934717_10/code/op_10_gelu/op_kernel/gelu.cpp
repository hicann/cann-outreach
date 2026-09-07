// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        length_ = length;
        inputGM_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x, length);
        outputGM_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output, length);

        // Keep a fixed vector length whose byte size is 32-byte aligned.
        pipe_.InitBuffer(inQueue_, 1, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outQueue_, 1, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(erfBuf_, TILE_LENGTH * sizeof(DT_INPUT_X));
    }
    __aicore__ inline void Process() {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t tileCount = (length_ + TILE_LENGTH - 1) / TILE_LENGTH;

        LocalTensor<DT_INPUT_X> erfValue = erfBuf_.Get<DT_INPUT_X>();

        // Assign independent tiles round-robin across AIVs. Each core owns
        // separate GM ranges, while VECIN/VECOUT queues order local work.
        for (uint32_t tile = blockIdx; tile < tileCount; tile += blockNum) {
            const uint32_t offset = tile * TILE_LENGTH;
            const uint32_t valid = (offset + TILE_LENGTH < length_) ?
                TILE_LENGTH : length_ - offset;
            const uint32_t calcLength =
                ((valid + VEC_ALIGN - 1) / VEC_ALIGN) * VEC_ALIGN;

            LocalTensor<DT_INPUT_X> x = inQueue_.AllocTensor<DT_INPUT_X>();
            if (valid == TILE_LENGTH) {
                // Every complete tile starts at a 32-byte aligned element
                // offset, so the fast DataCopy path is safe.
                DataCopy(x, inputGM_[offset], TILE_LENGTH);
            } else {
                // Only the final partial tile needs alignment-independent GM
                // access; it is bounded by one tile and does not dominate.
                Duplicate(x, static_cast<DT_INPUT_X>(0), calcLength);
                for (uint32_t i = 0; i < valid; ++i) {
                    x.SetValue(i, inputGM_.GetValue(offset + i));
                }
            }
            inQueue_.EnQue(x);
            x = inQueue_.DeQue<DT_INPUT_X>();

            // PyTorch F.gelu's default approximate="none" form uses erf.
            Muls(erfValue, x, static_cast<DT_INPUT_X>(0.7071067811865475f), calcLength);
            // Keep source and destination tensors distinct for binary/unary
            // vector instructions; this is supported across CANN releases.
            LocalTensor<DT_INPUT_X> result = outQueue_.AllocTensor<DT_INPUT_X>();
            Erf(result, erfValue, calcLength);
            Adds(erfValue, result, static_cast<DT_INPUT_X>(1.0f), calcLength);
            Mul(result, x, erfValue, calcLength);
            Muls(result, result, static_cast<DT_INPUT_X>(0.5f), calcLength);

            outQueue_.EnQue(result);
            result = outQueue_.DeQue<DT_INPUT_X>();
            if (valid == TILE_LENGTH) {
                DataCopy(outputGM_[offset], result, TILE_LENGTH);
            } else {
                for (uint32_t i = 0; i < valid; ++i) {
                    outputGM_.SetValue(offset + i, result.GetValue(i));
                }
            }
            outQueue_.FreeTensor(result);
            inQueue_.FreeTensor(x);
        }
    }
private:
    // 1024 elements amortize queue and instruction setup while keeping the
    // three UB buffers comfortably below typical Ascend 910B UB capacity.
    static constexpr uint32_t TILE_LENGTH = 1024;
    static constexpr uint32_t VEC_ALIGN = sizeof(DT_INPUT_X) == 2 ? 16 : 8;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> erfBuf_;
    GlobalTensor<DT_INPUT_X> inputGM_;
    GlobalTensor<DT_INPUT_X> outputGM_;
    uint32_t length_ = 0;

};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}