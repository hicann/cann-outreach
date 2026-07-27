#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t USE_CORE_NUM = 8;   // 使用的AI Core数量，需与op_host中SetBlockDim(8)一致
constexpr int32_t BUFFER_NUM = 2;     // DoubleBuffer深度
constexpr int32_t TILE_NUM = 8;       // 每个核内部切分的tile数量
constexpr int32_t DATA_ALIGNMENT = USE_CORE_NUM * TILE_NUM * BUFFER_NUM;

// 用模板支持float16/float32两种数据类型（由构建系统根据算子原型自动实例化DTYPE_X/Y/Z）
template<typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        // Host Tiling会拒绝未按核数、Tile数和DoubleBuffer深度对齐的输入。
        this->isValid = totalLength > 0 && totalLength % DATA_ALIGNMENT == 0;
        if (!this->isValid) {
            return;
        }

        // 每个核负责的数据长度
        this->blockLength = totalLength / USE_CORE_NUM;
        // 每个tile的实际长度
        this->tileLength = this->blockLength / TILE_NUM / BUFFER_NUM;

        // 根据当前核的BlockIdx计算该核负责的数据起始地址
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 为输入/输出队列申请DoubleBuffer缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));
    }

    __aicore__ inline void Process()
    {
        if (!this->isValid) {
            return;
        }

        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
        // 核心计算：逐元素相减 z = x - y
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<TYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
    bool isValid;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
