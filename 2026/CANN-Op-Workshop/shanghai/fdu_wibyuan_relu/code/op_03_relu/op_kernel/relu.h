/*!
 * \file relu.h
 * \brief Relu 算子 Kernel 实现
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "relu_tiling_data.h"

using namespace AscendC;

namespace NsRelu {

constexpr int32_t BUFFER_NUM = 2; // 开启 DoubleBuffer 机制提升性能

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    // 初始化：从 tiling 数据中获取总长度并完成多核分配与队列初始化
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ReluTilingData* tilingData) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        ASSERT(tilingData != nullptr && "tiling data is null!");

        uint32_t totalLength = static_cast<uint32_t>(tilingData->totalNum);
        uint32_t blockFactor = static_cast<uint32_t>(tilingData->blockFactor);
        uint32_t ubFactor = static_cast<uint32_t>(tilingData->ubFactor);

        uint32_t coreNum = GetBlockNum();
        uint32_t coreId = GetBlockIdx();

        // 32 字节对齐元素数（float 为 8，half 为 16）
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);

        // 计算本核处理的元素范围（已经按 ALIGN_NUM 对齐）
        uint32_t startBlock = coreId * blockFactor;
        if (startBlock >= totalLength) {
            this->coreLength = 0;
            this->tileNum = 0;
            return;
        }

        uint32_t currentCoreLength = blockFactor;
        if (startBlock + currentCoreLength > totalLength) {
            currentCoreLength = totalLength - startBlock;
        }
        this->coreLength = currentCoreLength;

        // 单块按 ubFactor 个元素分块（已对齐）
        this->tileSize = ubFactor;
        this->tileNum = (this->coreLength + this->tileSize - 1) / this->tileSize;

        // 绑定全局内存 GM
        xGm.SetGlobalBuffer((__gm__ T*)x + startBlock, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startBlock, this->coreLength);

        // 初始化流水线 Queue
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileSize * sizeof(T));
    }

    // 流水线驱动
    __aicore__ inline void Process() {
        if (coreLength == 0) return;

        for (int32_t i = 0; i < tileNum; i++) {
            uint32_t actualSize = tileSize;
            if (i == tileNum - 1) {
                actualSize = coreLength - i * tileSize;
            }
            constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);
            uint32_t copySize = ((actualSize + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            CopyIn(i, copySize);
            Compute(copySize);
            CopyOut(i, copySize);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t copySize) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * tileSize], copySize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t computeSize) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // 必须显式指明 AscendC::Relu，消除与本类名 Relu 的歧义
        AscendC::Relu(yLocal, xLocal, computeSize);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t copySize) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * tileSize], yLocal, copySize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> xGm, yGm;

    uint32_t coreLength;
    uint32_t tileNum;
    uint32_t tileSize;
};

} // namespace NsRelu

#endif // RELU_H