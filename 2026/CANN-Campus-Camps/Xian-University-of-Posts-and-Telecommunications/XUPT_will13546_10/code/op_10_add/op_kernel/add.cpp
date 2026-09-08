// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

constexpr int32_t BUFFER_NUM = 2;          // 双缓冲
constexpr uint32_t BLOCK_BYTES = 32;       // 搬运对齐单位：32字节

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddTilingData &tiling) {
        uint32_t block_idx = AscendC::GetBlockIdx();
        uint32_t block_num = AscendC::GetBlockNum();
        if (block_num == 0) {
            block_num = 1;
        }

        // 按核均分，余数分给前面若干个核，避免越界
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

        // 根据Host侧传入的块数计算实际单次搬运/计算的元素数
        uint32_t raw_tile_length = (blockLength_ + tiling.tileNum - 1) / tiling.tileNum;
        tileLength_ = ((raw_tile_length + align_elements - 1) / align_elements) * align_elements;
        if (tileLength_ == 0) {
            tileLength_ = align_elements;
        }
        tileNum_ = (blockLength_ + tileLength_ - 1) / tileLength_;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, blockLength_);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, blockLength_);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, blockLength_);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength_ * sizeof(DT_X));
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
        AscendC::LocalTensor<DT_X> y_local = inQueueY.template AllocTensor<DT_X>();

        uint32_t copy_bytes = current_length * sizeof(DT_X);
        AscendC::DataCopyParams copy_params{1, static_cast<uint16_t>(copy_bytes), 0, 0};
        AscendC::DataCopyPadParams pad_params{false, 0, 0, 0};

        AscendC::DataCopyPad(x_local, xGm[start], copy_params, pad_params);
        AscendC::DataCopyPad(y_local, yGm[start], copy_params, pad_params);

        inQueueX.template EnQue<DT_X>(x_local);
        inQueueY.template EnQue<DT_X>(y_local);
    }

    __aicore__ inline void Compute(uint32_t current_length) {
        AscendC::LocalTensor<DT_X> x_local = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> y_local = inQueueY.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> z_local = outQueueZ.template AllocTensor<DT_X>();

        AscendC::Add(z_local, x_local, y_local, static_cast<int32_t>(current_length));

        outQueueZ.template EnQue<DT_X>(z_local);
        inQueueX.FreeTensor(x_local);
        inQueueY.FreeTensor(y_local);
    }

    __aicore__ inline void CopyOut(uint32_t start, uint32_t current_length) {
        AscendC::LocalTensor<DT_X> z_local = outQueueZ.template DeQue<DT_X>();

        AscendC::DataCopyParams copy_params{1, static_cast<uint16_t>(current_length * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPad(zGm[start], z_local, copy_params);

        outQueueZ.FreeTensor(z_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 1;
    uint32_t tileNum_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}