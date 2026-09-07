// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileNum, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;

        // 当前核的起始偏移和实际处理长度（尾核可能不足blockLength）
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t startOffset = coreIdx * blockLength;
        currentLength = 0;
        if (startOffset < totalLength) {
            uint32_t remain = totalLength - startOffset;
            currentLength = (remain < blockLength) ? remain : blockLength;
        }

        // 循环次数（按tileLength切分，最后一块可能不满）
        loopCount = 0;
        if (currentLength > 0 && tileLength > 0) {
            loopCount = (currentLength + tileLength - 1) / tileLength;
            if (static_cast<uint32_t>(loopCount) > tileNum) loopCount = tileNum;
        }

        // 设置全局内存张量（从startOffset开始，长度为currentLength）
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, currentLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, currentLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + startOffset, currentLength);

        // 初始化UB缓冲区（每个tile三份，双缓冲）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (loopCount == 0) return;

        // 双缓冲流水：先预取第一块
        CopyIn(0);
        for (int32_t i = 0; i < loopCount; i++) {
            Compute(i);
            CopyOut(i);
            if (i != loopCount - 1) CopyIn(i + 1);  // 预取下一块
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t offset = static_cast<uint32_t>(progress) * tileLength;
        // 实际拷贝长度（最后一块可能不足tileLength）
        uint32_t copyLen = tileLength;
        if (offset + copyLen > currentLength) {
            copyLen = currentLength - offset;
        }
        curTileLen = copyLen;  // 供Compute和CopyOut使用

        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], copyLen);
        AscendC::DataCopy(yLocal, yGm[offset], copyLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, curTileLen);  // 使用实际长度

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t offset = static_cast<uint32_t>(progress) * tileLength;
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, curTileLen);  // 使用实际长度
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t currentLength;
    int32_t loopCount;
    uint32_t curTileLen;  // 当前 tile 的实际元素数
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