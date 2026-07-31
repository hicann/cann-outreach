// TilingKey模板定义的头文件
// ASCENDC_TPL_ARGS_DECL: host + kernel 共用声明
// ASCENDC_TPL_SEL: 在此定义以触发构建系统生成 MseLossTilingData class
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MseLoss,
    ASCENDC_TPL_DATATYPE_DECL(DT_PREDICT, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PREDICT, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);
