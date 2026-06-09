/**
 * Copyright (c) 2025. Ascend C abs operator (float16).
 *
 * Kernel implementation: y = |x| for half-precision floating point.
 * Supports shapes [1,128], [4,2048], [32,4096] on Ascend 910B3.
 *
 * Tiling strategy: each block processes a chunk of the input,
 * further split into tiles fitting the Unified Buffer (UB).
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;       // Double buffering
constexpr int32_t DEFAULT_TILE_LEN = 2048;  // Default tile length (elements)

/**
 * Ascend C kernel class for y = abs(x) on float16 data.
 */
class KernelAbs {
public:
    __aicore__ inline KernelAbs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
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

        // Allocate ping-pong buffers in UB
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLen * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLen * sizeof(half));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (this->blockLen + this->tileLen - 1) / this->tileLen;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curTile = this->tileLen;
            // Last tile may be smaller
            if (i == loopCount - 1 && this->blockLen % this->tileLen != 0) {
                curTile = this->blockLen - i * this->tileLen;
            }

            // ---- Pipe: load input from GM -> UB ----
            LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            DataCopy(xLocal, xGm[i * this->tileLen], curTile);

            // ---- Compute: xLocal = |xLocal| ----
            // Abs in Ascend C works element-wise on LocalTensor
            Abs(xLocal, xLocal, curTile);

            // ---- Pipe: store result UB -> GM ----
            LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
            DataCopy(yLocal, xLocal, curTile);
            DataCopy(yGm[i * this->tileLen], yLocal, curTile);

            // Free tensors for reuse
            inQueueX.FreeTensor(xLocal);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t blockStart;
    uint32_t blockLen;
    uint32_t tileLen;
};

// ---------------------------------------------------------------------------
// Tiling data structure passed from host to kernel via TVM global workspace
// ---------------------------------------------------------------------------
struct AbsCustomTilingData {
    uint32_t totalLength;   // Total number of float16 elements
    uint32_t tileLen;       // Elements per tile
};

// ---------------------------------------------------------------------------
// Kernel entry point (called by the Ascend C runtime)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void abs_custom(
    GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tvmGlobalWorkspace) {
    GET_TILING_DATA(tilingData, totalLength);
    // TODO: use workspace if needed for intermediate storage
    (void)workspace;

    KernelAbs op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileLen);
    op.Process();
}
