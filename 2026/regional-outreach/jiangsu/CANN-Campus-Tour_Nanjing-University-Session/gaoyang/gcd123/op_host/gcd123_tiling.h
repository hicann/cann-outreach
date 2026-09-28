/**
 * @file gcd123_tiling.h
 *
 * TilingData for the Gcd123 operator.
 *
 * This header is shared by the host side (op_host/gcd123.cpp) and the
 * kernel side (op_kernel/gcd123.cpp). The host fills it in TilingFunc,
 * the kernel reads it back through GET_TILING_DATA.
 *
 * All shapes are 4-dimensional [N4, N3, N2, N1] in ND format.
 * selfShape/otherShape must satisfy the standard broadcast relation,
 * outShape is the broadcast result of selfShape and otherShape.
 */
#ifndef GCD123_TILING_H
#define GCD123_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Gcd123TilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, selfN4);
TILING_DATA_FIELD_DEF(uint32_t, selfN3);
TILING_DATA_FIELD_DEF(uint32_t, selfN2);
TILING_DATA_FIELD_DEF(uint32_t, selfN1);
TILING_DATA_FIELD_DEF(uint32_t, otherN4);
TILING_DATA_FIELD_DEF(uint32_t, otherN3);
TILING_DATA_FIELD_DEF(uint32_t, otherN2);
TILING_DATA_FIELD_DEF(uint32_t, otherN1);
TILING_DATA_FIELD_DEF(uint32_t, outN4);
TILING_DATA_FIELD_DEF(uint32_t, outN3);
TILING_DATA_FIELD_DEF(uint32_t, outN2);
TILING_DATA_FIELD_DEF(uint32_t, outN1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd123, Gcd123TilingData)

} // namespace optiling

#endif // GCD123_TILING_H
