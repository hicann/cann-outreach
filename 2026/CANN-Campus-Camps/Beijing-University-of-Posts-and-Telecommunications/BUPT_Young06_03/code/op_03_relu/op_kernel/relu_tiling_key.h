/*! 
 * \file relu_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __RELU_TILING_KEY_H__
#define __RELU_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Relu,
    ASCENDC_TPL_DATATYPE_DECL(
        DT_X,
        C_DT_FLOAT,
        C_DT_FLOAT16,
        ASCENDC_TPL_INPUT(0)
    )
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16)
    )
);

#endif