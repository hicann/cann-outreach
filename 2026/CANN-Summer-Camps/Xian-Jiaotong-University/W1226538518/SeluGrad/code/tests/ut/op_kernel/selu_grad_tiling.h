/*!
 * \file selu_grad_tiling.h
 * \brief SeluGrad UT 兼容层: tiling 数据初始化
 */

#ifndef _I_SELUGRAD_TILING_H_
#define _I_SELUGRAD_TILING_H_

#include <cstdint>
#include <cstring>
#include "../../../op_kernel/selu_grad_tiling_data.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "graph/c_types.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

inline void InitTilingData(uint8_t *tiling, SeluGradTilingData *constData)
{
    memcpy(constData, tiling, sizeof(SeluGradTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    SeluGradTilingData tilingData;          \
    InitTilingData(tilingArg, &tilingData)

#endif
