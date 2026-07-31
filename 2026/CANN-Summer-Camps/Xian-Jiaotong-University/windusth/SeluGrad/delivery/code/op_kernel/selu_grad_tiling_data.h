/*!
 * \file selu_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SELUGRAD_TILING_DATA_H_
#define _SELUGRAD_TILING_DATA_H_

struct SeluGradTilingData {
    int64_t totalNum = 0;     // Input/output element count.
    int64_t blockFactor = 1;  // Elements assigned to each AIV core.
    int64_t ubFactor = 0;     // Reserved for vectorized implementations.
};
#endif
