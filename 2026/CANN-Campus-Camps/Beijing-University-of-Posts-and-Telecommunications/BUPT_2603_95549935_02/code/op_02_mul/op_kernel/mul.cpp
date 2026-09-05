#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &td) {
        this->length = td.length;
        this->blockNum = td.blockNum;
        this->blockLen = td.blockLen;
        this->tileLen = td.tileLen;

        uint32_t core = AscendC::GetBlockIdx();
        this->offset = this->blockLen * core;
        this->myLen = (core == this->blockNum - 1) ? (this->length - this->offset) : this->blockLen;

        this->xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->offset, this->myLen);
        this->yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->offset, this->myLen);
        this->zGm.SetGlobalBuffer((__gm__ DT_X*)z + this->offset, this->myLen);

        pipe.InitBuffer(xQue, BUFFER_NUM, this->tileLen * sizeof(DT_X));
        pipe.InitBuffer(yQue, BUFFER_NUM, this->tileLen * sizeof(DT_X));
        pipe.InitBuffer(zQue, BUFFER_NUM, this->tileLen * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t rounds = (this->myLen + this->tileLen - 1) / this->tileLen;
        for (uint32_t i = 0; i < rounds; i++) {
            uint32_t len = this->myLen - i * this->tileLen;
            if (len > this->tileLen) {
                len = this->tileLen;
            }
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len) {
        AscendC::LocalTensor<DT_X> xLocal = xQue.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQue.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLen], len);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLen], len);
        xQue.EnQue(xLocal);
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        AscendC::LocalTensor<DT_X> zLocal = zQue.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> xLocal = xQue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQue.DeQue<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, len);
        zQue.EnQue(zLocal);
        xQue.FreeTensor<DT_X>(xLocal);
        yQue.FreeTensor<DT_X>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len) {
        AscendC::LocalTensor<DT_X> zLocal = zQue.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLen], zLocal, len);
        zQue.FreeTensor<DT_X>(zLocal);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue, yQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQue;

    uint32_t length;
    uint32_t blockNum;
    uint32_t blockLen;
    uint32_t tileLen;
    uint32_t offset;
    uint32_t myLen;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
