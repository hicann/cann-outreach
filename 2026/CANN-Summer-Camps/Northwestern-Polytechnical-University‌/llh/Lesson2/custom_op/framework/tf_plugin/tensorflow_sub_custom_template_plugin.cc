/* -------------------------------------------------------------------------
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 * ------------------------------------------------------------------------- */

#include "register/register.h"

namespace domi {
// register op info to GE for TensorFlow
REGISTER_CUSTOM_OP("SubCustomTemplate")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SubCustomTemplate")
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
