/**
 * Copyright (c) 2025. Ascend C GCD operator (float16, broadcast, 4D ND).
 *
 * Kernel implementation: out = gcd(self, other) for half-precision.
 * Supports 4D broadcast semantics: shape [N4,N3,N2,N1], Format=ND.
 *
 * GCD algorithm (Euclidean):
 *   while |b| > eps: (a, b) = (b, a mod b); return |a|
 *
 * Tiling strategy:
 *   The output tensor is the broadcast result of self and other.
 *   Each block processes a chunk of output elements.
 *   For each output element, the corresponding self/other indices
 *   are computed via strides (0 for broadcast dimensions).
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;           // Double buffering
constexpr int32_t DEFAULT_TILE_LEN = 2048;  // Default tile length (elements)
constexpr half   GCD_EPS = (half)1e-4f;     // Convergence threshold
constexpr int32_t GCD_MAX_ITER = 50;        // Max Euclidean iterations

/**
 * Ascend C kernel class for out = gcd(self, other) on float16 data.
 *
 * Supports broadcasting: self/other strides with value 0 indicate
 * a dimension of size 1 (broadcast), so the index always reads
 * element 0 along that axis.
 */
class KernelGcd {
public:
    __aicore__ inline KernelGcd() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                uint32_t totalLength, uint32_t tileLen,
                                const int32_t *selfStrides,
                                const int32_t *otherStrides,
                                const int32_t *outStrides) {
        uint32_t blockNum = GetBlockNum();
        ASSERT(blockNum != 0 && "block dim cannot be zero");

        // ---- Distribute output elements evenly across blocks ----
        uint32_t baseLen = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        uint32_t blockIdx  = GetBlockIdx();

        this->blockStart = blockIdx * baseLen + (blockIdx < remainder ? blockIdx : remainder);
        this->blockLen   = baseLen + (blockIdx < remainder ? 1 : 0);

        // Tile length shall not exceed block length
        if (tileLen == 0 || tileLen > this->blockLen) {
            tileLen = this->blockLen;
        }
        // Align tile length to 32 bytes (16 half elements)
        tileLen = (tileLen / 16) * 16;
        if (tileLen == 0) tileLen = 16;
        this->tileLen = tileLen;

        // Point global tensors (whole arrays; indexing is done via strides)
        selfGm.SetGlobalBuffer((__gm__ half *)self);
        otherGm.SetGlobalBuffer((__gm__ half *)other);
        outGm.SetGlobalBuffer((__gm__ half *)out);

        // Copy stride info for broadcast mapping
        for (int32_t d = 0; d < 4; d++) {
            this->selfStrides[d]  = selfStrides[d];
            this->otherStrides[d] = otherStrides[d];
            this->outStrides[d]   = outStrides[d];
        }

        // Allocate ping-pong buffers for output (contiguous writes)
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLen * sizeof(half));
    }

    /**
     * Convert a flat output index to a flat self/other index using
     * the broadcast-aware strides.
     *
     * outStrides:  [N3*N2*N1, N2*N1, N1, 1]  (output tensor)
     * selfStrides: same pattern, but with 0 where selfShape[d]==1 (broadcast)
     */
    __aicore__ inline uint32_t mapIndex(uint32_t flatIdx,
                                         const int32_t *strides) const {
        uint32_t n4 = flatIdx / (uint32_t)outStrides[0];
        uint32_t r  = flatIdx % (uint32_t)outStrides[0];

        uint32_t n3 = r / (uint32_t)outStrides[1];
        r = r % (uint32_t)outStrides[1];

        uint32_t n2 = r / (uint32_t)outStrides[2];
        uint32_t n1 = r % (uint32_t)outStrides[2];

        return (uint32_t)(n4 * (uint32_t)strides[0] +
                          n3 * (uint32_t)strides[1] +
                          n2 * (uint32_t)strides[2] +
                          n1 * (uint32_t)strides[3]);
    }

    /**
     * Scalar GCD computation for float16 using Euclidean algorithm.
     *
     *   gcd(a, b) = gcd(|a|, |b|)
     *   while |b| > eps: (a, b) = (b, a mod b)
     *
     * For float16, modulo is computed as: r = a - floor(a/b) * b
     */
    __aicore__ inline half gcd_half(half a, half b) const {
        // Take absolute values
        a = a < (half)0.0 ? -a : a;
        b = b < (half)0.0 ? -b : b;

        // Edge cases
        if (a == (half)0.0) return b;
        if (b == (half)0.0) return a;

        // Ensure a >= b
        if (a < b) {
            half t = a; a = b; b = t;
        }

        // Euclidean algorithm
        for (int32_t i = 0; i < GCD_MAX_ITER; i++) {
            if (b <= GCD_EPS) break;

            // Compute remainder: r = a - floor(a/b) * b
            half ratio = a / b;
            // floor for positive ratio: truncate toward zero then adjust
            int32_t ratioInt = (int32_t)ratio;
            if ((half)ratioInt > ratio) {
                ratioInt = ratioInt - 1;
            }
            half r = a - (half)ratioInt * b;

            // If remainder is effectively zero, we're done
            if (r < (half)0.0) r = -r;
            if (r <= GCD_EPS) break;

            a = b;
            b = r;
        }

        return a < (half)0.0 ? -a : a;
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (this->blockLen + this->tileLen - 1) / this->tileLen;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curTile = this->tileLen;
            if (i == loopCount - 1 && this->blockLen % this->tileLen != 0) {
                curTile = this->blockLen - i * this->tileLen;
            }

            uint32_t outBase = this->blockStart + i * this->tileLen;

            // Allocate output buffer in UB
            LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            // For each element in this tile, compute GCD with broadcast lookup
            for (uint32_t j = 0; j < curTile; j++) {
                uint32_t outIdx = outBase + j;

                // Map output index to self and other indices via broadcast strides
                uint32_t selfIdx = mapIndex(outIdx, this->selfStrides);
                uint32_t otherIdx = mapIndex(outIdx, this->otherStrides);

                // Scalar loads from GM
                half selfVal = selfGm(selfIdx);
                half otherVal = otherGm(otherIdx);

                // Compute GCD
                half result = gcd_half(selfVal, otherVal);

                // Store to UB buffer
                yLocal(j) = result;
            }

            // Vectorized write: UB -> GM (contiguous output)
            DataCopy(outGm[outBase], yLocal, curTile);

            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> selfGm;
    GlobalTensor<half> otherGm;
    GlobalTensor<half> outGm;
    uint32_t blockStart;
    uint32_t blockLen;
    uint32_t tileLen;
    int32_t selfStrides[4];
    int32_t otherStrides[4];
    int32_t outStrides[4];
};

// ---------------------------------------------------------------------------
// Tiling data structure passed from host to kernel
// ---------------------------------------------------------------------------
struct GcdCustomTilingData {
    uint32_t totalLength;       // Total output elements (after broadcast)
    uint32_t tileLen;           // Elements per tile
    int32_t selfStrides[4];     // Self strides (0 for broadcast dims)
    int32_t otherStrides[4];    // Other strides (0 for broadcast dims)
    int32_t outStrides[4];      // Output strides for 4D->1D mapping
};

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void gcd_custom(
    GM_ADDR self, GM_ADDR other, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tvmGlobalWorkspace) {
    GET_TILING_DATA(tilingData, totalLength);
    (void)workspace;

    KernelGcd op;
    op.Init(self, other, out,
            tilingData.totalLength, tilingData.tileLen,
            tilingData.selfStrides, tilingData.otherStrides,
            tilingData.outStrides);
    op.Process();
}
