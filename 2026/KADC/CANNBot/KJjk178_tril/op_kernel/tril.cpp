#include "kernel_operator.h"

#include "tril_tiling.h"

#include "tiling_key_tril.h"

constexpr uint32_t ALIGN_NUM = 16;

template <class DT_X>
class KernelTril {
public:
    __aicore__ inline KernelTril() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t M, uint32_t N,
                                uint32_t batchSize, int32_t diagonal,
                                uint32_t totalRows, uint32_t rowsPerCore, uint32_t tailRows) {
        M_ = M;
        N_ = N;
        diagonal_ = diagonal;
        alignedN_ = (N + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        if (alignedN_ == 0) alignedN_ = ALIGN_NUM;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockIdx < blockNum - 1) {
            myRows_ = rowsPerCore;
        } else {
            myRows_ = tailRows;
        }
        myStartRow_ = blockIdx * rowsPerCore;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);
        uint32_t bufferBytes = alignedN_ * sizeof(DT_X);
        pipe.InitBuffer(inQueue_, 1, bufferBytes);
        pipe.InitBuffer(outQueue_, 1, bufferBytes);
    }

    __aicore__ inline void Process() {
        if (myRows_ == 0 || N_ == 0) return;
        for (uint32_t r = 0; r < myRows_; r++) {
            uint32_t globalRow = myStartRow_ + r;
            uint32_t matrixIdx = globalRow / M_;
            uint32_t rowInMatrix = globalRow % M_;
            uint32_t rowOffset = matrixIdx * M_ * N_ + rowInMatrix * N_;

            int32_t validCols = (int32_t)rowInMatrix + diagonal_ + 1;
            if (validCols < 0) validCols = 0;
            if (validCols > (int32_t)N_) validCols = (int32_t)N_;

            if (validCols == 0) {
                WriteZeros(rowOffset, N_);
            } else {
                CopyRow(rowOffset);
                if (validCols < (int32_t)N_) {
                    WriteZeros(rowOffset + validCols, N_ - validCols);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyRow(uint32_t rowOffset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        AscendC::DataCopyPad(xLocal, xGm_[rowOffset],
            {1, (uint16_t)(N_ * sizeof(DT_X)), 0, 0},
            {false, 0, 0, 0});
        inQueue_.EnQue(xLocal);

        xLocal = inQueue_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        AscendC::Adds(yLocal, xLocal, (DT_X)0, alignedN_);
        inQueue_.FreeTensor(xLocal);
        outQueue_.EnQue<DT_X>(yLocal);

        yLocal = outQueue_.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[rowOffset], yLocal,
            {1, (uint16_t)(N_ * sizeof(DT_X)), 0, 0});
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void WriteZeros(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> zeroLocal = outQueue_.AllocTensor<DT_X>();
        AscendC::Duplicate(zeroLocal, (DT_X)0, alignedN_);
        outQueue_.EnQue<DT_X>(zeroLocal);

        zeroLocal = outQueue_.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[offset], zeroLocal,
            {1, (uint16_t)(count * sizeof(DT_X)),
             (uint16_t)(alignedN_ * sizeof(DT_X)), 0});
        outQueue_.FreeTensor(zeroLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    uint32_t M_, N_, alignedN_, myRows_, myStartRow_;
    int32_t diagonal_;
};

template <typename DT_X>
 __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TrilTilingData);
    GET_TILING_DATA_WITH_STRUCT(TrilTilingData, tiling_data, tiling);
    KernelTril<DT_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.M, tiling_data.N,
            tiling_data.batchSize, tiling_data.diagonal,
            tiling_data.totalRows, tiling_data.rowsPerCore, tiling_data.tailRows);
    op.Process();
}
