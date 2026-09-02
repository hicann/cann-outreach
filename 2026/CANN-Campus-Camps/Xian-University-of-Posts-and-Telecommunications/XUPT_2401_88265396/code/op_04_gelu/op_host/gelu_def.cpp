/*!
 * \file gelu_def.cpp
 * \brief Gelu 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name)
    {
    this->Input("input_x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
} // namespace ops
