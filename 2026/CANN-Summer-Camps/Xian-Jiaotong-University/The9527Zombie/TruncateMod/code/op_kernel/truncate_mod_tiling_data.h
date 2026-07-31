/*!
 * \file truncate_mod_tiling_data.h
 * \brief TruncateMod tiling data structure, shared between host tiling and device kernel.
 *
 * TruncateMod computes the truncated remainder element-wise:
 *     y = x1 - trunc(x1 / x2) * x2
 * where trunc() rounds toward zero, so the remainder keeps the sign of x1
 * (identical to numpy/torch fmod semantics).
 *
 * The kernel supports NumPy-style broadcasting between x1 and x2. The output
 * shape is the broadcast of the two inputs. For every output dimension we keep,
 * for each input, the element stride to walk that input in the output's
 * coordinate space (a stride of 0 means the input is broadcast along that dim).
 */
#ifndef _TRUNCATEMOD_TILING_DATA_H_
#define _TRUNCATEMOD_TILING_DATA_H_

#include <cstdint>

// schMode (tiling key) selects the compute dtype on the device side.
#ifndef TRUNCATE_MOD_SCH_FP16
#define TRUNCATE_MOD_SCH_FP16 0 // half (float16)
#endif
#ifndef TRUNCATE_MOD_SCH_FP32
#define TRUNCATE_MOD_SCH_FP32 1 // float (float32)
#endif
#ifndef TRUNCATE_MOD_SCH_BF16
#define TRUNCATE_MOD_SCH_BF16 2 // bfloat16
#endif
#ifndef TRUNCATE_MOD_SCH_INT32
#define TRUNCATE_MOD_SCH_INT32 3 // int32
#endif
#ifndef TRUNCATE_MOD_SCH_INT8
#define TRUNCATE_MOD_SCH_INT8 4 // int8
#endif
#ifndef TRUNCATE_MOD_SCH_UINT8
#define TRUNCATE_MOD_SCH_UINT8 5 // uint8
#endif

// Maximum rank handled by the broadcast walker.
#ifndef TRUNCATE_MOD_MAX_DIM
#define TRUNCATE_MOD_MAX_DIM 8
#endif

struct TruncateModTilingData {
    uint64_t coreNum = 1;         // AI cores actually launched
    uint64_t totalCount = 0;      // total output element count
    uint64_t tileCount = 0;       // max elements processed per UB tile
    uint64_t perCoreCount = 0;    // base output elements per core (block aligned)
    uint64_t tailCoreNum = 0;     // number of leading cores taking one extra block
    uint64_t lastCoreCount = 0;   // output elements handled by the last core
    uint64_t blockElem = 0;       // elements per 32B block for the current dtype
    uint32_t bufferNum = 1;       // 1 or 2 (double buffer)
    uint32_t dimNum = 0;          // effective output rank
    uint32_t x1SameShape = 1;     // x1 shape == output shape (no broadcast for x1)
    uint32_t x2SameShape = 1;     // x2 shape == output shape (no broadcast for x2)
    uint32_t x1Scalar = 0;        // x1 is a single-element tensor
    uint32_t x2Scalar = 0;        // x2 is a single-element tensor
    uint32_t reserved = 0;        // keep 8-byte alignment for the arrays below
    uint64_t outShape[TRUNCATE_MOD_MAX_DIM] = {0};
    uint64_t x1Stride[TRUNCATE_MOD_MAX_DIM] = {0};
    uint64_t x2Stride[TRUNCATE_MOD_MAX_DIM] = {0};
};
#endif // _TRUNCATEMOD_TILING_DATA_H_
