// Kernel侧核函数实现
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

using namespace AscendC;

constexpr int32_t TILE_LENGTH = 4096;  // 每段 UB 处理元素数
constexpr int32_t BUFFER_NUM = 2;      // 双缓冲

template <class DT_X>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->totalLength_ = static_cast<int32_t>(length);
        int32_t blockDim = static_cast<int32_t>(GetBlockNum());
        int32_t coreIdx = static_cast<int32_t>(GetBlockIdx());
        // 每核元素数：均分上取整后 32 对齐，保证 GM 偏移 32 字节对齐
        int32_t blockLength = this->totalLength_ / blockDim;
        if (blockLength * blockDim < this->totalLength_) {
            blockLength += 1;
        }
        blockLength = (blockLength + 31) / 32 * 32;
        this->blockLength_ = blockLength;
        // 本核实际处理范围：尾部钳制，越界核跳过
        int32_t offset = coreIdx * blockLength;
        this->processLength_ = (offset >= this->totalLength_)
                                   ? 0
                                   : ((this->totalLength_ - offset) < blockLength
                                          ? (this->totalLength_ - offset)
                                          : blockLength);
        this->loopNum_ = (this->processLength_ + TILE_LENGTH - 1) / TILE_LENGTH;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->processLength_);
        inputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->processLength_);
        outputGMZ.SetGlobalBuffer((__gm__ DT_X*)z + offset, this->processLength_);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inputQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (int32_t progress = 0; progress < this->loopNum_; progress++) {
            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    __aicore__ inline int32_t CurNum(int32_t progress) const {
        int32_t remain = this->processLength_ - progress * TILE_LENGTH;
        return (remain < TILE_LENGTH) ? remain : TILE_LENGTH;
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inputQueueY.AllocTensor<DT_X>();
        int32_t cur = CurNum(progress);
        DataCopy(xLocal, inputGMX[progress * TILE_LENGTH], cur);
        DataCopy(yLocal, inputGMY[progress * TILE_LENGTH], cur);
        inputQueueX.EnQue(xLocal);
        inputQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inputQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outputQueueZ.AllocTensor<DT_X>();
        AscendC::Add(zLocal, xLocal, yLocal, CurNum(progress));
        inputQueueX.FreeTensor(xLocal);
        inputQueueY.FreeTensor(yLocal);
        outputQueueZ.EnQue(zLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DT_X> zLocal = outputQueueZ.DeQue<DT_X>();
        DataCopy(outputGMZ[progress * TILE_LENGTH], zLocal, CurNum(progress));
        outputQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> inputGMY;
    GlobalTensor<DT_X> outputGMZ;
    int32_t totalLength_ = 0;    // 总元素数量
    int32_t blockLength_ = 0;    // 每核元素数量（32 对齐）
    int32_t processLength_ = 0;  // 本核实际处理元素数量（尾部钳制）
    int32_t loopNum_ = 0;        // 本核段循环次数
};

template <typename DT_X>
 __global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
