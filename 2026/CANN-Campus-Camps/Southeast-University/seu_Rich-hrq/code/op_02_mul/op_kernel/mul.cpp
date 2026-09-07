// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;      // 多块路径的队列缓冲个数（double buffer）
constexpr uint32_t ONE_BLK_SIZE = 32;  // DataCopy 要求 32 字节对齐

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileNum, uint32_t tileLength) {
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        // 本核数据范围 [offset, offset + curBlockLength)，尾核可能不足 blockLength
        uint32_t offset = AscendC::GetBlockIdx() * blockLength;
        uint32_t remain = (totalLength > offset) ? (totalLength - offset) : 0;
        this->blockLength = (remain < blockLength) ? remain : blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->blockLength);
        if (tileNum == 1) {
            // 单块快速路径：TBuf 直配，免队列 Alloc/EnQue/DeQue/Free 开销
            pipe.InitBuffer(bufX, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(bufY, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(bufZ, this->tileLength * sizeof(DT_X));
        } else {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        }
    }
    __aicore__ inline void Process() {
        if (this->tileNum == 1) {
            ProcessSingleTile();
            return;
        }
        for (uint32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    // ===== 单块快速路径：TBuf + HardEvent 显式同步 =====
    // 每核每张量一次 DMA，无缓冲复用，无需 MTE3_MTE2 回收等待；
    // kernel 退出屏障保证最后的 MTE3 写回排空。
    __aicore__ inline void ProcessSingleTile() {
        AscendC::LocalTensor<DT_X> xLocal = bufX.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = bufY.Get<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = bufZ.Get<DT_X>();
        // MTE2: GM -> UB
        CopyGmToUb(xLocal, xGm, this->blockLength);
        CopyGmToUb(yLocal, yGm, this->blockLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // Vector: 等数据就绪后计算
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Mul(zLocal, xLocal, yLocal, this->blockLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        // MTE3: 等计算完成后写回
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        CopyUbToGm(zGm, zLocal, this->blockLength);
    }
    // ===== 多块通用路径 =====
    // 第 progress 块的实际长度（核内尾块可能小于 tileLength）
    __aicore__ inline uint32_t CurTileLength(uint32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t remain = (this->blockLength > offset) ? (this->blockLength - offset) : 0;
        return (remain < this->tileLength) ? remain : this->tileLength;
    }
    // GM -> UB：长度 32B 对齐走 DataCopy；非对齐（仅总长度非 32B 对齐时出现）走 DataCopyPad
    __aicore__ inline void CopyGmToUb(const AscendC::LocalTensor<DT_X> &dst,
                                      const AscendC::GlobalTensor<DT_X> &src, uint32_t length) {
        if ((length * sizeof(DT_X)) % ONE_BLK_SIZE == 0) {
            AscendC::DataCopy(dst, src, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(dst, src, copyParams, padParams);
        }
    }
    // UB -> GM：同上（UB->GM 的 DataCopyPad 自动丢弃对齐填充数据）
    __aicore__ inline void CopyUbToGm(const AscendC::GlobalTensor<DT_X> &dst,
                                      const AscendC::LocalTensor<DT_X> &src, uint32_t length) {
        if ((length * sizeof(DT_X)) % ONE_BLK_SIZE == 0) {
            AscendC::DataCopy(dst, src, length);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(dst, src, copyParams);
        }
    }
    __aicore__ inline void CopyIn(uint32_t progress) {
        uint32_t length = CurTileLength(progress);
        uint32_t offset = progress * this->tileLength;
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        CopyGmToUb(xLocal, xGm[offset], length);
        CopyGmToUb(yLocal, yGm[offset], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t progress) {
        uint32_t length = CurTileLength(progress);
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress) {
        uint32_t length = CurTileLength(progress);
        uint32_t offset = progress * this->tileLength;
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        CopyUbToGm(zGm[offset], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    AscendC::TPipe pipe;
    // 单块快速路径（tileNum == 1）
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufZ;
    // 多块通用路径（tileNum > 1）：depth=1（官方推荐），双缓冲由 InitBuffer num=2 控制
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;  // 本核实际处理的元素个数
    uint32_t tileNum;      // 核内分块个数
    uint32_t tileLength;   // 每块元素个数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength,
            tiling_data.tileNum, tiling_data.tileLength);
    op.Process();
}
