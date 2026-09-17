// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MAX_TILE_LENGTH = 512;  // Erf 需要额外 UB

template <class DT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData &tiling) {
        uint32_t block_idx = AscendC::GetBlockIdx();
        uint32_t block_num = AscendC::GetBlockNum();
        if (block_num == 0) {
            block_num = 1;
        }

        uint32_t base = tiling.totalLength / block_num;
        uint32_t remainder = tiling.totalLength % block_num;
        blockLength_ = base + (block_idx < remainder ? 1 : 0);
        uint32_t offset = block_idx * base + (block_idx < remainder ? block_idx : remainder);

        if (blockLength_ == 0 || tiling.tileNum == 0) {
            tileLength_ = 1;
            tileNum_ = 0;
            return;
        }

        uint32_t align_elements = BLOCK_BYTES / sizeof(DT_X);
        if (align_elements == 0) {
            align_elements = 1;
        }

        uint32_t raw_tile_length = (blockLength_ + tiling.tileNum - 1) / tiling.tileNum;
        tileLength_ = ((raw_tile_length + align_elements - 1) / align_elements) * align_elements;
        if (tileLength_ == 0) {
            tileLength_ = align_elements;
        }
        if (tileLength_ > MAX_TILE_LENGTH) {
            tileLength_ = (MAX_TILE_LENGTH / align_elements) * align_elements;
            if (tileLength_ == 0) {
                tileLength_ = align_elements;
            }
        }
        tileNum_ = (blockLength_ + tileLength_ - 1) / tileLength_;

        xGm.SetGlobalBuffer((__gm__ DT_X *)input_x + offset, blockLength_);
        yGm.SetGlobalBuffer((__gm__ DT_X *)output + offset, blockLength_);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0 || tileNum_ == 0) {
            return;
        }
        for (uint32_t i = 0; i < tileNum_; ++i) {
            uint32_t start = i * tileLength_;
            uint32_t current_length = CurrentLength(start);
            if (current_length == 0) {
                continue;
            }
            CopyIn(start, current_length);
            Compute(current_length);
            CopyOut(start, current_length);
        }
    }

private:
    __aicore__ inline uint32_t CurrentLength(uint32_t start) const {
        if (start >= blockLength_) {
            return 0;
        }
        uint32_t remain = blockLength_ - start;
        return remain < tileLength_ ? remain : tileLength_;
    }

    __aicore__ inline void CopyIn(uint32_t start, uint32_t current_length) {
        AscendC::LocalTensor<DT_X> x_local = inQueueX.template AllocTensor<DT_X>();

        uint32_t copy_bytes = current_length * sizeof(DT_X);
        AscendC::DataCopyParams copy_params{1, static_cast<uint16_t>(copy_bytes), 0, 0};
        AscendC::DataCopyPadParams pad_params{false, 0, 0, 0};

        AscendC::DataCopyPad(x_local, xGm[start], copy_params, pad_params);
        inQueueX.template EnQue<DT_X>(x_local);
    }

    __aicore__ inline void Compute(uint32_t current_length) {
        AscendC::LocalTensor<DT_X> x_local = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> y_local = outQueueY.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> tmp_local = tmpBuf.template Get<DT_X>();

        // y = 0.5 * x * (1 + erf(x / sqrt(2)))
        AscendC::Muls(tmp_local, x_local, static_cast<DT_X>(0.7071067811865475244f),
                      static_cast<int32_t>(current_length));
        AscendC::Erf(y_local, tmp_local, static_cast<int32_t>(current_length));
        AscendC::Adds(y_local, y_local, static_cast<DT_X>(1.0f),
                      static_cast<int32_t>(current_length));
        AscendC::Mul(y_local, x_local, y_local, static_cast<int32_t>(current_length));
        AscendC::Muls(y_local, y_local, static_cast<DT_X>(0.5f),
                      static_cast<int32_t>(current_length));

        outQueueY.template EnQue<DT_X>(y_local);
        inQueueX.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOut(uint32_t start, uint32_t current_length) {
        AscendC::LocalTensor<DT_X> y_local = outQueueY.template DeQue<DT_X>();

        AscendC::DataCopyParams copy_params{1, static_cast<uint16_t>(current_length * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPad(yGm[start], y_local, copy_params);

        outQueueY.FreeTensor(y_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 1;
    uint32_t tileNum_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}