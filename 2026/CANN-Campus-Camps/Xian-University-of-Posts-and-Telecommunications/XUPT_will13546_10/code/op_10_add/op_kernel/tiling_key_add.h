#ifndef TILING_KEY_ADD_H
#define TILING_KEY_ADD_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Add,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);

#endif  // TILING_KEY_ADD_H