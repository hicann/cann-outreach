// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength) {
        this->blockLength = blockLength;
        pipe.InitBuffer(xQue, 1, blockLength * sizeof(DT_X));
        pipe.InitBuffer(yQue, 1, blockLength * sizeof(DT_X));
        pipe.InitBuffer(zQue, 1, blockLength * sizeof(DT_X));

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + AscendC::GetBlockIdx() * blockLength, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + AscendC::GetBlockIdx() * blockLength, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + AscendC::GetBlockIdx() * blockLength, blockLength);
    }
    __aicore__ inline void Process() {
        LocalTensor<DT_X> xLocal = xQue.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = yQue.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm, blockLength);
        DataCopy(yLocal, yGm, blockLength);
        xQue.EnQue(xLocal);
        yQue.EnQue(yLocal);

        LocalTensor<DT_X> xBuf = xQue.DeQue<DT_X>();
        LocalTensor<DT_X> yBuf = yQue.DeQue<DT_X>();
        LocalTensor<DT_X> zBuf = zQue.AllocTensor<DT_X>();
        Mul(zBuf, xBuf, yBuf, blockLength);
        xQue.FreeTensor(xBuf);
        yQue.FreeTensor(yBuf);
        zQue.EnQue(zBuf);

        LocalTensor<DT_X> zResult = zQue.DeQue<DT_X>();
        DataCopy(zGm, zResult, blockLength);
        zQue.FreeTensor(zResult);
    }
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> xQue;
    TQue<TPosition::VECIN, 1> yQue;
    TQue<TPosition::VECOUT, 1> zQue;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t blockLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.blockLength);
    op.Process();
}
