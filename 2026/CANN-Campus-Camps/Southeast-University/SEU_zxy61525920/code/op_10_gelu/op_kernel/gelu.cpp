// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr uint32_t GELU_TILE_LENGTH = 1024;
constexpr float GELU_INV_SQRT_TWO = 0.70710678118654752440f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length) {

        const uint32_t block_num = GetBlockNum();
        const uint32_t block_idx = GetBlockIdx();

        // 将全部元素均匀分配给各个AI Core
        const uint32_t base = length / block_num;
        const uint32_t remainder = length % block_num;

        block_length_ =
            base + (block_idx < remainder ? 1 : 0);

        block_offset_ =
            block_idx * base +
            (block_idx < remainder ? block_idx : remainder);

        input_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) +
                block_offset_,
            block_length_);

        output_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(output) +
                block_offset_,
            block_length_);

        pipe_.InitBuffer(
            input_queue_,
            1,
            GELU_TILE_LENGTH * sizeof(DT_INPUT_X));

        pipe_.InitBuffer(
            output_queue_,
            1,
            GELU_TILE_LENGTH * sizeof(DT_INPUT_X));

        pipe_.InitBuffer(
            tmp_buffer_,
            GELU_TILE_LENGTH * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t offset = 0;
             offset < block_length_;
             offset += GELU_TILE_LENGTH) {

            const uint32_t remaining = block_length_ - offset;

            const uint32_t count =
                remaining > GELU_TILE_LENGTH
                    ? GELU_TILE_LENGTH
                    : remaining;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t count) {

        LocalTensor<DT_INPUT_X> input_local =
            input_queue_.AllocTensor<DT_INPUT_X>();

        DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(
                count * sizeof(DT_INPUT_X)),
            0,
            0,
            0};

        DataCopyPadExtParams<DT_INPUT_X> pad_params{
            false,
            0,
            0,
            0};

        DataCopyPad(
            input_local,
            input_gm_[offset],
            copy_params,
            pad_params);

        input_queue_.EnQue(input_local);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_INPUT_X> input_local =
            input_queue_.DeQue<DT_INPUT_X>();

        LocalTensor<DT_INPUT_X> output_local =
            output_queue_.AllocTensor<DT_INPUT_X>();

        LocalTensor<DT_INPUT_X> tmp_local =
            tmp_buffer_.Get<DT_INPUT_X>();

        // GELU(x) = 0.5 * x *
        //           (1 + erf(x / sqrt(2)))
        Muls(
            tmp_local,
            input_local,
            static_cast<DT_INPUT_X>(GELU_INV_SQRT_TWO),
            count);

        Erf<DT_INPUT_X, false>(
            output_local,
            tmp_local,
            count);

        Adds(
            output_local,
            output_local,
            static_cast<DT_INPUT_X>(1.0f),
            count);

        Mul(
            output_local,
            input_local,
            output_local,
            count);

        Muls(
            output_local,
            output_local,
            static_cast<DT_INPUT_X>(0.5f),
            count);

        output_queue_.EnQue(output_local);
        input_queue_.FreeTensor(input_local);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t count) {

        LocalTensor<DT_INPUT_X> output_local =
            output_queue_.DeQue<DT_INPUT_X>();

        DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(
                count * sizeof(DT_INPUT_X)),
            0,
            0,
            0};

        DataCopyPad(
            output_gm_[offset],
            output_local,
            copy_params);

        output_queue_.FreeTensor(output_local);
    }

private:
    TPipe pipe_;

    TQue<QuePosition::VECIN, 1> input_queue_;
    TQue<QuePosition::VECOUT, 1> output_queue_;
    TBuf<QuePosition::VECCALC> tmp_buffer_;

    GlobalTensor<DT_INPUT_X> input_gm_;
    GlobalTensor<DT_INPUT_X> output_gm_;

    uint32_t block_length_ = 0;
    uint32_t block_offset_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling) {

    REGISTER_TILING_DEFAULT(GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tiling_data,
        tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}