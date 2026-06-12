/**
 * Copyright (c) 2025. Ascend C add operator (float16).
 *
 * Kernel implementation: z = x + y for half-precision floating point.
 * Supports 2D shapes [N2, N1] on Ascend 910B3.
 *
 * Tiling strategy: each block processes a chunk of the inputs,
 * further split into tiles fitting the Unified Buffer (UB).
 * Double-buffering hides data-movement latency.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;            // Double buffering
constexpr int32_t DEFAULT_TILE_LEN = 2048;   // Default tile length (elements)

/**
 * Ascend C kernel class for z = x + y on float16 data.
 *
 * Each block processes a contiguous chunk of the flat tensor,
 * subdivided into tiles that fit in UB.
 */
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLen) {
        uint32_t blockNum = GetBlockNum();
        ASSERT(blockNum != 0 && "block dim cannot be zero");

        // ---- Distribute data evenly across blocks ----
        uint32_t baseLen = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        uint32_t blockIdx  = GetBlockIdx();

        // Blocks with index < remainder get one extra element
        this->blockStart = blockIdx * baseLen + (blockIdx < remainder ? blockIdx : remainder);
        this->blockLen   = baseLen + (blockIdx < remainder ? 1 : 0);

        // Tile length shall not exceed block length
        if (tileLen == 0 || tileLen > this->blockLen) {
            tileLen = this->blockLen;
        }
        // Align tile length to 32 bytes (16 half elements) for efficiency
        tileLen = (tileLen / 16) * 16;
        if (tileLen == 0) tileLen = 16;
        this->tileLen = tileLen;

        // Point global tensors to this block's region
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockStart);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockStart);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockStart);

        // Allocate ping-pong buffers in UB
        // 2 inputs × double-buffer + 1 output × double-buffer = 6 buffers
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLen * sizeof(half));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLen * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLen * sizeof(half));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (this->blockLen + this->tileLen - 1) / this->tileLen;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curTile = this->tileLen;
            // Last tile may be smaller
            if (i == loopCount - 1 && this->blockLen % this->tileLen != 0) {
                curTile = this->blockLen - i * this->tileLen;
            }

            // ---- Pipe: load inputs from GM -> UB ----
            LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
            DataCopy(xLocal, xGm[i * this->tileLen], curTile);
            DataCopy(yLocal, yGm[i * this->tileLen], curTile);

            // ---- Compute: zLocal = xLocal + yLocal ----
            // Add in Ascend C works element-wise on LocalTensors.
            // We use xLocal as the destination to save one buffer:
            //   xLocal = xLocal + yLocal
            Add(xLocal, xLocal, yLocal, curTile);

            // ---- Pipe: store result UB -> GM ----
            LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
            DataCopy(zLocal, xLocal, curTile);
            DataCopy(zGm[i * this->tileLen], zLocal, curTile);

            // Free tensors for reuse
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
            outQueueZ.FreeTensor(zLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockStart;
    uint32_t blockLen;
    uint32_t tileLen;
};

// ---------------------------------------------------------------------------
// Tiling data structure passed from host to kernel via TVM global workspace
// ---------------------------------------------------------------------------
struct AddCustomTilingData {
    uint32_t totalLength;   // Total number of float16 elements
    uint32_t tileLen;       // Elements per tile
};

// ---------------------------------------------------------------------------
// Kernel entry point (called by the Ascend C runtime)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tvmGlobalWorkspace) {
    GET_TILING_DATA(tilingData, totalLength);
    (void)workspace;

    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLen);
    op.Process();
}
