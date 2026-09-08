/*!
 * \file square_tiling_key.h
 * \brief Schedule mode 0: FP16; schedule mode 1: FP32.
 */
#ifndef SQUARE_TILING_KEY_H
#define SQUARE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SQUARE_TPL_SCH_MODE_0 0
#define SQUARE_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Square,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST,
                         SQUARE_TPL_SCH_MODE_0, SQUARE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                        SQUARE_TPL_SCH_MODE_0, SQUARE_TPL_SCH_MODE_1)));

#endif // SQUARE_TILING_KEY_H
