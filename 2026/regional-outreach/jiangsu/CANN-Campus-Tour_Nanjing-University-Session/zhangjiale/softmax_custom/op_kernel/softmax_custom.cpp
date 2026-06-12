/**
 * Copyright (c) 2025. Ascend C softmax operator (float32).
 *
 * Implementation: softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m)), m = max(x)
 *
 * Shape: [128,64,32], dtype: float32, axis=-1, Format=ND.
 *   - Total softmax rows: 128 * 64 = 8192
 *   - Each row: 32 float32 elements (128 bytes)
 *   - Each row is an independent softmax computation
 *
 * === API 验证记录 ===
 * 以下 API 的用法均对照 abs_ascend 已验证代码确认:
 *   DataCopy(dst, src, cnt)     → abs: DataCopy(xLocal, xGm[...], curTile) ✓
 *   GlobalTensor::operator[]    → abs: xGm[i * this->tileLen] ✓
 *   SetGlobalBuffer(ptr)        → abs: xGm.SetGlobalBuffer((__gm__ half*)x + ...) ✓
 *   Abs(dst, src, len)          → abs: Abs(xLocal, xLocal, curTile) ✓
 *   GetBlockNum/GetBlockIdx     → abs: 使用中 ✓
 *   Que::AllocTensor/FreeTensor → abs: 使用中 ✓
 *
 * 以下 API 由 AscendC 标准库提供，签名与 Abs 一致:
 *   Exp(dst, src, len)          → dst[i] = exp(src[i])
 *   ReduceMax(dst, src, len)    → dst[0] = max(src[0..len-1])
 *   ReduceSum(dst, src, len)    → dst[0] = sum(src[0..len-1])
 *   Muls(dst, src, scalar, len) → dst[i] = src[i] * scalar
 *   Adds(dst, src, scalar, len) → dst[i] = src[i] + scalar
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // Double buffering slots
constexpr int32_t ROW_LEN   = 32;   // Elements per row (softmax axis dim)

/**
 * Softmax kernel: one row at a time, double buffered.
 */
class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalRows) {
        uint32_t blockNum = GetBlockNum();
        ASSERT(blockNum != 0 && "block dim cannot be zero");

        // ---- Distribute rows evenly across blocks ----
        uint32_t baseRows   = totalRows / blockNum;
        uint32_t remainder  = totalRows % blockNum;
        uint32_t blockIdx   = GetBlockIdx();

        // Blocks 0..remainder-1 each get one extra row
        this->rowStart = blockIdx * baseRows +
                         (blockIdx < remainder ? blockIdx : remainder);
        this->rowCount = baseRows + (blockIdx < remainder ? 1 : 0);

        // ---- Point global tensors to this block's row range ----
        xGm.SetGlobalBuffer((__gm__ float *)x + this->rowStart * ROW_LEN);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->rowStart * ROW_LEN);

        // ---- Allocate ping-pong queues (one row per slot) ----
        uint32_t rowBytes = ROW_LEN * sizeof(float);  // 128 B
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, rowBytes);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowBytes);

        // ---- Allocate reduction temps (1 float each) ----
        pipe.InitBuffer(maxBuf, sizeof(float));  // row max
        pipe.InitBuffer(sumBuf, sizeof(float));  // row sum
    }

    __aicore__ inline void Process() {
        for (uint32_t r = 0; r < this->rowCount; r++) {
            // ── Load row r from GM → UB ───────────────────────────
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            // Ref[abs]: DataCopy(xLocal, xGm[i * this->tileLen], curTile);
            DataCopy(xLocal, xGm[r * ROW_LEN], ROW_LEN);

            // ── Step 1: m = max(x[0..31]) ─────────────────────────
            LocalTensor<float> maxLocal = maxBuf.Get<float>();
            // ReduceMax -> dst[0] = max(src[0..len-1])
            ReduceMax(maxLocal, xLocal, ROW_LEN);
            float maxVal = maxLocal.GetValue(0);

            // ── Step 2: x = x - m  (numerical stability) ──────────
            // Adds(dst, src, scalar, len) -> dst[i] = src[i] + scalar
            Adds(xLocal, xLocal, -maxVal, ROW_LEN);

            // ── Step 3: x = exp(x) ─────────────────────────────────
            // Exp(dst, src, len) -> dst[i] = exp(src[i])
            Exp(xLocal, xLocal, ROW_LEN);

            // ── Step 4: s = sum(x) ─────────────────────────────────
            LocalTensor<float> sumLocal = sumBuf.Get<float>();
            // ReduceSum -> dst[0] = sum(src[0..len-1])
            ReduceSum(sumLocal, xLocal, ROW_LEN);
            float sumVal = sumLocal.GetValue(0);

            // ── Step 5: x = x / s = x * (1/s) ─────────────────────
            float invSum = 1.0f / sumVal;
            // Muls(dst, src, scalar, len) -> dst[i] = src[i] * scalar
            Muls(xLocal, xLocal, invSum, ROW_LEN);

            // ── Store row r from UB → GM ──────────────────────────
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            // Ref[abs]: DataCopy(yLocal, xLocal, curTile);
            DataCopy(yLocal, xLocal, ROW_LEN);
            // Ref[abs]: DataCopy(yGm[i * this->tileLen], yLocal, curTile);
            DataCopy(yGm[r * ROW_LEN], yLocal, ROW_LEN);

            // Free for reuse (double-buffer slot returns to pool)
            inQueueX.FreeTensor(xLocal);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;   // GM → UB
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;  // UB → GM
    TBuf<TPosition::VECCALC> maxBuf;   // 1 × float: row maximum
    TBuf<TPosition::VECCALC> sumBuf;   // 1 × float: row sum
    GlobalTensor<float> xGm;           // Input  in GM
    GlobalTensor<float> yGm;           // Output in GM
    uint32_t rowStart;                 // First row for this block
    uint32_t rowCount;                 // Rows owned by this block
};

// ---------------------------------------------------------------------------
// Tiling data (passed from host → kernel via tvmGlobalWorkspace)
// ---------------------------------------------------------------------------
struct SoftmaxCustomTilingData {
    uint32_t totalRows;
};

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void softmax_custom(
    GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tvmGlobalWorkspace) {
    GET_TILING_DATA(tilingData, totalRows);
    (void)workspace;

    KernelSoftmax op;
    op.Init(x, y, tilingData.totalRows);
    op.Process();
}
