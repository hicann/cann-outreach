/*!
 * \file sync_batch_norm_backward_reduce.cpp
 * \brief SyncBatchNormBackwardReduce 算子 kernel 入口
 */

#include "sync_batch_norm_backward_reduce.h"

enum class SyncBatchNormBackwardReduceTilingKey : uint32_t
{
    TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_0 = 0,
    TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_1 = 1,
    TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void sync_batch_norm_backward_reduce(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean, GM_ADDR invert_std, GM_ADDR sum_dy_xmu, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBatchNormBackwardReduceTilingData);
    GET_TILING_DATA_WITH_STRUCT(SyncBatchNormBackwardReduceTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SyncBatchNormBackwardReduceTilingKey::TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_0)) {
        NsSyncBatchNormBackwardReduce::SyncBatchNormBackwardReduce<half> op;
        op.Init(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SyncBatchNormBackwardReduceTilingKey::TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_1)) {
        NsSyncBatchNormBackwardReduce::SyncBatchNormBackwardReduce<float> op;
        op.Init(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SyncBatchNormBackwardReduceTilingKey::TILING_KEY_SYNCBATCHNORMBACKWARDREDUCE_MODE_2)) {
        NsSyncBatchNormBackwardReduce::SyncBatchNormBackwardReduce<bfloat16_t> op;
        op.Init(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, &tilingData);
        op.Process();
    }
}
