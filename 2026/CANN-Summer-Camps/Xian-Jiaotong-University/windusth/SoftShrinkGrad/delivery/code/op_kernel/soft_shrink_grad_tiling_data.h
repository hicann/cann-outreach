/*!
 * \file soft_shrink_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SOFTSHRINKGRAD_TILING_DATA_H_
#define _SOFTSHRINKGRAD_TILING_DATA_H_

struct SoftShrinkGradTilingData {
    int64_t totalNum = 0;     // Input/output element count.
    int64_t blockFactor = 1;  // Elements assigned to each AIV core.
    int64_t ubFactor = 0;     // Reserved for vectorized implementations.
    float lambd = 0.5f;
};
#endif
