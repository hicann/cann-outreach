#include "register/register.h"

namespace domi {
// register op info to GE
REGISTER_CUSTOM_OP("AddCustomTemplate")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("AddCustomTemplate")      // name in tf module
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
