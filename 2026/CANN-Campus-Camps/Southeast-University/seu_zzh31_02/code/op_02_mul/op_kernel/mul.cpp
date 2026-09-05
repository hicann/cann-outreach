// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 逐元素矢量乘法 z = x * y
// 多核一维连续均分：每个逻辑AI Core处理一段连续元素 [blockStart, blockStart+currentBlockLen)
// 数据流：GM(x/y) -> DataCopyPad -> UB(VECIN 队列) -> 向量Mul -> UB(VECOUT) -> DataCopyPad -> GM(z)
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength) {
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        // 当前逻辑核的连续区间：blockStart 为元素偏移，currentBlockLen 为实际元素数（尾块钳制）
        int64_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockStart = static_cast<uint32_t>(blockIdx) * blockLength;
        uint32_t currentBlockLen = blockLength;
        if (blockStart >= totalLength) {
            currentBlockLen = 0;
        } else if (blockStart + blockLength > totalLength) {
            currentBlockLen = totalLength - blockStart;
        }
        this->blockStart = blockStart;
        this->currentBlockLen = currentBlockLen;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockStart, currentBlockLen);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockStart, currentBlockLen);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockStart, currentBlockLen);

        // 单缓冲 depth=1；tileLength*sizeof(DT_X) 为 256B 对齐
        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t tileNum = currentBlockLen / tileLength;
        uint32_t tailElem = currentBlockLen - tileNum * tileLength;
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength);
            Compute(tileLength);
            CopyOut(i * tileLength, tileLength);
        }
        if (tailElem > 0) {
            CopyIn(tileNum * tileLength, tailElem);
            Compute(tailElem);
            CopyOut(tileNum * tileLength, tailElem);
        }
    }

private:
    // GM -> UB 搬入 x/y，统一用 DataCopyPad（非对齐尾块安全）
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], cp, pp);
        AscendC::DataCopyPad(yLocal, yGm[offset], cp, pp);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // UB 内向量 Mul：z = x * y（count 连续计算接口，非对齐尾块由硬件 Counter 处理）
    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, static_cast<int32_t>(count));
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // UB -> GM 搬出 z，统一用 DataCopyPad
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(zGm[offset], zLocal, cp);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t blockStart;
    uint32_t currentBlockLen;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}