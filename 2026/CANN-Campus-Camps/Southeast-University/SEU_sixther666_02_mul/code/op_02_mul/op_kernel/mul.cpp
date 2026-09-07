// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t VECTOR_REPEAT_BYTES = 256;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->blockLength = length / BLOCK_DIM;
        uint32_t blockOffset = AscendC::GetBlockIdx() * this->blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, this->blockLength);

        // 单核数据远小于UB容量，一次整块处理可消除循环和DoubleBuffer开销。
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm, this->blockLength);
        AscendC::DataCopy(yLocal, yGm, this->blockLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        constexpr uint64_t mask = VECTOR_REPEAT_BYTES / sizeof(DT_X);
        uint8_t repeatTime = static_cast<uint8_t>(this->blockLength * sizeof(DT_X) / VECTOR_REPEAT_BYTES);
        AscendC::Mul(zLocal, xLocal, yLocal, mask, repeatTime, {1, 1, 1, 8, 8, 8});

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm, zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}