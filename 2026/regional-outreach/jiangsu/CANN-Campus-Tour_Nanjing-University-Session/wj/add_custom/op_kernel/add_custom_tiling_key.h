/*!
 * \file add_custom_tiling_key.h
 * \brief add_custom tiling key declare
 *
 * add_custom 仅支持 float16，因此 tiling key 只有一个调度模式：
 *   - schMode 0: float16
 */

#ifndef __ADD_CUSTOM_TILING_KEY_H__
#define __ADD_CUSTOM_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(AddCustom,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0)));

#endif
