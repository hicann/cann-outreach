/*!
 * \file truncate_mod_tiling_key.h
 * \brief TruncateMod tiling-key template argument declaration.
 *
 * A single template argument "schMode" (3 bits) selects the compute dtype.
 * The host tiling picks the value from the input dtype and the device kernel
 * dispatches to the matching typed implementation.
 */
#ifndef __TRUNCATEMOD_TILING_KEY_H__
#define __TRUNCATEMOD_TILING_KEY_H__

#include "truncate_mod_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TruncateMod, ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, TRUNCATE_MOD_SCH_FP16,
                                                         TRUNCATE_MOD_SCH_FP32, TRUNCATE_MOD_SCH_BF16,
                                                         TRUNCATE_MOD_SCH_INT32, TRUNCATE_MOD_SCH_INT8,
                                                         TRUNCATE_MOD_SCH_UINT8));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TRUNCATE_MOD_SCH_FP16,
                                                          TRUNCATE_MOD_SCH_FP32, TRUNCATE_MOD_SCH_BF16,
                                                          TRUNCATE_MOD_SCH_INT32, TRUNCATE_MOD_SCH_INT8,
                                                          TRUNCATE_MOD_SCH_UINT8)));
#endif // __TRUNCATEMOD_TILING_KEY_H__
