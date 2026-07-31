/*!
 * \file sync_batch_norm_backward_reduce_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SYNCBATCHNORMBACKWARDREDUCE_TILING_DATA_H_
#define _SYNCBATCHNORMBACKWARDREDUCE_TILING_DATA_H_

struct SyncBatchNormBackwardReduceTilingData {
    int64_t totalNum = 0;     // Input/output element count.
    int64_t blockFactor = 1;  // Elements assigned to each AIV core.
    int64_t ubFactor = 0;     // Reserved for vectorized implementations.
};
#endif
