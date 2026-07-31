/*!
 * \file truncate_mod_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __TRUNCATEMOD_TILING_KEY_H__
#define __TRUNCATEMOD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TRUNCATEMOD_TPL_SCH_MODE_0 0
#define TRUNCATEMOD_TPL_SCH_MODE_1 1
#define TRUNCATEMOD_TPL_SCH_MODE_2 2

ASCENDC_TPL_ARGS_DECL(
    TruncateMod,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, TRUNCATEMOD_TPL_SCH_MODE_0, TRUNCATEMOD_TPL_SCH_MODE_1,
                          TRUNCATEMOD_TPL_SCH_MODE_2));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TRUNCATEMOD_TPL_SCH_MODE_0, TRUNCATEMOD_TPL_SCH_MODE_1,
                         TRUNCATEMOD_TPL_SCH_MODE_2)));

#endif
