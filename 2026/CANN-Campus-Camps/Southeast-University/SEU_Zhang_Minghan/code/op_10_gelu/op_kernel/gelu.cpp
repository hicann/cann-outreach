// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;  // 输入/输出队列均使用双缓冲

template <class DT_INPUT_X>
class KernelGelu {
public:
    using TYPE_X = DT_INPUT_X;
    using TYPE_Y = DT_INPUT_X;

    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x,
                                GM_ADDR output,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum) {
        // 当前核号
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        // 前 tailBlockNum 个核(大核)比其它核(小核)多承担一个 32B 数据块
        const bool isBigCore = (coreIdx < tailBlockNum);

        this->coreDataNum = isBigCore ? bigCoreDataNum : smallCoreDataNum;
        this->tileNum = isBigCore ? finalBigTileNum : finalSmallTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = isBigCore ? bigTailDataNum : smallTailDataNum;
        this->invSqrt2 = static_cast<TYPE_X>(0.7071067811865475f);  // 1 / sqrt(2)

        // 计算本核在输入/输出张量中的起始元素偏移: 大核连续排在最前, 小核紧随其后
        uint32_t startOffset = 0;
        if (isBigCore) {
            startOffset = bigCoreDataNum * coreIdx;
        } else {
            startOffset = bigCoreDataNum * tailBlockNum +
                          smallCoreDataNum * (coreIdx - tailBlockNum);
        }

        this->xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE_X *>(input_x) + startOffset, this->coreDataNum);
        this->yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE_Y *>(output) + startOffset, this->coreDataNum);

        // 为输入/输出队列申请双缓冲空间
        this->pipe.InitBuffer(this->inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe.InitBuffer(this->outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    }

    __aicore__ inline void Process() {
        // 每个核循环处理自己负责的连续数据, 最后一个 tile 长度可能小于 tileDataNum
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            const uint32_t curTileSize =
                (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            this->CopyIn(i, curTileSize);
            this->Compute(curTileSize);
            this->CopyOut(i, curTileSize);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curTileSize) {
        AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, this->xGm[progress * this->tileDataNum], curTileSize);
        this->inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curTileSize) {
        AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template AllocTensor<TYPE_Y>();

        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  (精确误差函数定义)
        AscendC::Muls(yLocal, xLocal, this->invSqrt2, curTileSize);  // y = x / sqrt(2)
        AscendC::Erf(yLocal, yLocal, curTileSize);                   // y = erf(x / sqrt(2))
        AscendC::Adds(yLocal, yLocal, static_cast<TYPE_Y>(1.0f),
                      curTileSize);  // y = 1 + erf(x / sqrt(2))
        AscendC::Muls(yLocal, yLocal, static_cast<TYPE_Y>(0.5f),
                      curTileSize);  // y = 0.5 * (1 + erf(x / sqrt(2)))
        AscendC::Mul(yLocal, yLocal, xLocal,
                     curTileSize);  // y = 0.5 * x * (1 + erf(x / sqrt(2)))

        this->outQueue.EnQue(yLocal);
        this->inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curTileSize) {
        AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template DeQue<TYPE_Y>();
        AscendC::DataCopy(this->yGm[progress * this->tileDataNum], yLocal, curTileSize);
        this->outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    TYPE_X invSqrt2;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x,
            output,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}