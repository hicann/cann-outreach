// Kernel侧核函数实现
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;   // 双缓冲
constexpr uint32_t TILE_LENGTH = 2048; // 每次流水迭代处理的元素个数（大 tile 减少流水切换开销）
constexpr uint32_t ALIGN_UNIT = 32;   // GM 偏移按 32 元素对齐（fp32/fp16 均满足 32 字节对齐）

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        this->length = length;
        // 多核切分：每核处理的元素数向上对齐到 32 的倍数，保证 GM 偏移 32 字节对齐
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockLength = (length + blockNum - 1) / blockNum;
        blockLength = ((blockLength + ALIGN_UNIT - 1) / ALIGN_UNIT) * ALIGN_UNIT;
        this->blockLength = blockLength;
        uint32_t offset = blockIdx * blockLength;
        this->coreLength = (offset >= length) ? 0
                              : ((length - offset) < blockLength ? (length - offset) : blockLength);

        inputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + offset, (int32_t)this->coreLength);
        outputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + offset, (int32_t)this->coreLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
    }
    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        uint32_t tileCount = (this->coreLength + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < tileCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 当前 tile 的实际元素数（最后一块可能不足 TILE_LENGTH）
    __aicore__ inline uint32_t CurLength(uint32_t idx) const {
        uint32_t remain = this->coreLength - idx * TILE_LENGTH;
        return (remain < TILE_LENGTH) ? remain : TILE_LENGTH;
    }

    __aicore__ inline void CopyIn(uint32_t idx) {
        uint32_t len = CurLength(idx);
        LocalTensor<DT_INPUT_X> xLocal = inQueue.AllocTensor<DT_INPUT_X>();
        DataCopy(xLocal, inputGlobal[idx * TILE_LENGTH], (int32_t)len);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t idx) {
        uint32_t len = CurLength(idx);
        LocalTensor<DT_INPUT_X> xLocal = inQueue.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> outLocal = outQueue.AllocTensor<DT_INPUT_X>();
        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        // 变换链 in-place 复用输出缓冲：省去一块中间 UB 缓冲，减少队列管理开销
        Muls(outLocal, xLocal, (DT_INPUT_X)0.7071067811865476f, (int32_t)len); // x / sqrt(2)
        Erf(outLocal, outLocal, (int32_t)len);                                  // erf(x / sqrt(2))
        Adds(outLocal, outLocal, (DT_INPUT_X)1.0f, (int32_t)len);               // 1 + erf(...)
        Muls(outLocal, outLocal, (DT_INPUT_X)0.5f, (int32_t)len);               // 0.5 * (1 + erf(...))
        Mul(outLocal, xLocal, outLocal, (int32_t)len);                          // x * 0.5 * (1 + erf(...))
        inQueue.FreeTensor(xLocal);
        outQueue.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        uint32_t len = CurLength(idx);
        LocalTensor<DT_INPUT_X> outLocal = outQueue.DeQue<DT_INPUT_X>();
        DataCopy(outputGlobal[idx * TILE_LENGTH], outLocal, (int32_t)len);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<DT_INPUT_X> inputGlobal;
    GlobalTensor<DT_INPUT_X> outputGlobal;
    uint32_t length;
    uint32_t blockLength;
    uint32_t coreLength;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}