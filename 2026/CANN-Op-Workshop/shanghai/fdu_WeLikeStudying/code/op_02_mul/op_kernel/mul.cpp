// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        uint32_t coreNum = GetBlockNum();
        uint32_t coreId = GetBlockIdx();

        // 32 字节对应的元素个数（float 为 8，half 为 16）
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);

        // 1. 将总元素向上对齐到 32 字节，计算总 block 数
        uint32_t totalAlignLength = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

        // 2. 保证每个核处理的数据量都是 ALIGN_NUM 的整数倍，确保每个核的起始地址 100% 32B 对齐
        uint32_t totalBlocks = totalAlignLength / ALIGN_NUM;
        uint32_t blocksPerCore = (totalBlocks + coreNum - 1) / coreNum;

        uint32_t startBlock = coreId * blocksPerCore;
        if (startBlock >= totalBlocks) {
            // 当前核没有分配到任务
            this->coreLength = 0;
            this->tileNum = 0;
            return;
        }

        uint32_t currentCoreBlocks = blocksPerCore;
        if (startBlock + currentCoreBlocks > totalBlocks) {
            currentCoreBlocks = totalBlocks - startBlock;
        }

        uint32_t coreOffset = startBlock * ALIGN_NUM;
        uint32_t alignCoreLength = currentCoreBlocks * ALIGN_NUM;

        // 当前核负责的有效真实长度（防止超过总长度 totalLength）
        if (coreOffset + alignCoreLength > totalLength) {
            this->coreLength = totalLength - coreOffset;
        } else {
            this->coreLength = alignCoreLength;
        }

        // 3. 内部 Tile 切分，必须是 ALIGN_NUM 的整数倍
        this->tileSize = 512; // 512 是 8 和 16 的公倍数
        this->tileNum = (this->coreLength + this->tileSize - 1) / this->tileSize;

        // 4. 绑定 GlobalBuffer
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, this->coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + coreOffset, this->coreLength);

        // 5. 初始化队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileSize * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileSize * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) return;

        for (int32_t i = 0; i < tileNum; i++) {
            uint32_t actualSize = tileSize;
            if (i == tileNum - 1) {
                actualSize = coreLength - i * tileSize;
            }

            constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
            // 搬运长度向上对齐
            uint32_t copySize = ((actualSize + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            CopyIn(i, copySize);
            Compute(copySize); // 计算也按 copySize 进行，避免脏数据残留
            CopyOut(i, copySize);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t copySize) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm[progress * tileSize], copySize);
        DataCopy(yLocal, yGm[progress * tileSize], copySize);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t computeSize) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, computeSize);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t copySize) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[progress * tileSize], zLocal, copySize);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t coreLength;
    uint32_t tileNum;
    uint32_t tileSize;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}