/**
 * @file inplace_rsqrt.h
 * @brief In-place reciprocal square root for Ascend C LocalTensor.
 *
 * Formula : self[i] = 1 / sqrt(self[i])
 * Dtype   : half (float16)
 * Format  : ND
 * Impl    : AscendC::Rsqrt (vectorized vrsqrt), no scalar element loop.
 */
#ifndef INPLACE_RSQRT_H
#define INPLACE_RSQRT_H

#include "kernel_operator.h"

template <typename T = half>
__aicore__ inline void InplaceRsqrt(AscendC::LocalTensor<T> &self, const int32_t count)
{
    AscendC::Rsqrt(self, self, count);
}

#endif // INPLACE_RSQRT_H
