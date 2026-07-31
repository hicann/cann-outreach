/*!
 * \file truncate_mod_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __TRUNCATEMOD_TILING_KEY_H__
#define __TRUNCATEMOD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TruncateMod,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X,
        C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, C_DT_INT32, C_DT_INT8, C_DT_UINT8,
        ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

#define TMM_SEL(dt) \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, dt), \
                         ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1))

ASCENDC_TPL_SEL(
    TMM_SEL(C_DT_FLOAT),
    TMM_SEL(C_DT_FLOAT16),
    TMM_SEL(C_DT_BF16),
    TMM_SEL(C_DT_INT32),
    TMM_SEL(C_DT_INT8),
    TMM_SEL(C_DT_UINT8)
);

#undef TMM_SEL

#endif
